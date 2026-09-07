/*
 * GHOST v2 — CSI Injection over USB SERIAL (no Wi-Fi).
 *
 * The host writes injection lines to the ESP32 over the same USB serial cable,
 * and the ESP32 echoes back a standard CSI_DATA line for each one. Routing is
 * by USB port (one board per cable), so there is no Wi-Fi, no IP, no node id.
 *
 *   host --INJ line-->  ESP32 (this fw)  --CSI_DATA line-->  host
 *
 * Input line format (host -> ESP), on UART0 at the console baud:
 *   INJ,<seq>,<v0>,<v1>,...,<v127>\n         (128 interleaved I,Q int16 values)
 *
 * Output (ESP -> host). Two wire formats, selected at compile time:
 *
 *   CSI_OUTPUT_BINARY = 1 (default): a binary "G2" frame per injection —
 *     ["G2"][ver][node][flags][rsv][seq:u32be][num_sub:u16be][I0,Q0,...:i16be]
 *     ≈4x less UART traffic than the ASCII line, so the link stops dropping
 *     packets. Matches v2/transport/frame.py exactly (BinarySerialSource reads it).
 *
 *   CSI_OUTPUT_BINARY = 0: the legacy standard ESP32 CSI_DATA CSV line:
 *     CSI_DATA,<seq>,<mac>,...,"[v0,v1,...,v127]"\n
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"

#define UART_PORT     UART_NUM_0
#define RX_BUF_SIZE   4096
#define INJ_LINE_MAX      4096
#define NUM_VALS      128        /* 64 subcarriers * (I,Q) */

/* Wire format: 1 = binary G2 frames (default), 0 = legacy ASCII CSI_DATA line. */
#ifndef CSI_OUTPUT_BINARY
#define CSI_OUTPUT_BINARY 1
#endif

/* Must match v2/transport/frame.py. */
#define G2_VERSION      1
#define G2_NODE_ID      0        /* routing is by USB port; host ignores this */
#define G2_HEADER_SIZE  12
#define G2_NUM_SUB      (NUM_VALS / 2)

#if CSI_OUTPUT_BINARY

static inline void put_u16_be(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)(v & 0xff);
}

static inline void put_u32_be(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v & 0xff);
}

/* Emit one binary G2 frame (big-endian header + interleaved i16be I/Q). */
static void emit_csi(uint32_t seq, const int16_t *iq, int n_vals)
{
    static uint8_t buf[G2_HEADER_SIZE + NUM_VALS * 2];
    buf[0] = 'G';
    buf[1] = '2';
    buf[2] = G2_VERSION;
    buf[3] = G2_NODE_ID;
    buf[4] = 0;                          /* flags: 0 (not a calibration frame) */
    buf[5] = 0;                          /* reserved */
    put_u32_be(&buf[6], seq);
    put_u16_be(&buf[10], G2_NUM_SUB);
    for (int i = 0; i < n_vals; i++) {
        put_u16_be(&buf[G2_HEADER_SIZE + i * 2], (uint16_t)iq[i]);
    }
    uart_write_bytes(UART_PORT, (const char *)buf, G2_HEADER_SIZE + n_vals * 2);
}

#else  /* legacy ASCII CSI_DATA line */

/* MAC stamped into the ASCII line (matches config_v2.TX_MAC). */
static const uint8_t TX_MAC[6] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};

/* Format one CSI_DATA line (same 24-field layout as the Wi-Fi firmware/mock). */
static void emit_csi(uint32_t seq, const int16_t *iq, int n_vals)
{
    int64_t ts = esp_timer_get_time();
    printf("CSI_DATA,%u,%02x:%02x:%02x:%02x:%02x:%02x,"
           "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%d,%d,%d,%d,%d",
           (unsigned)seq,
           TX_MAC[0], TX_MAC[1], TX_MAC[2], TX_MAC[3], TX_MAC[4], TX_MAC[5],
           -45, 11, 1, 7, 1, 0, 1, 0, 0, 0, 0, -95, 0, 6, 0,
           (long long)ts, 0, n_vals, 1, n_vals, 0);
    printf(",\"[");
    for (int i = 0; i < n_vals; i++) {
        printf(i ? ",%d" : "%d", iq[i]);
    }
    printf("]\"\n");
}

#endif  /* CSI_OUTPUT_BINARY */

/* Parse one "INJ,seq,v0,...,v127" line and emit the matching CSI_DATA line. */
static void handle_line(char *line)
{
    if (strncmp(line, "INJ,", 4) != 0) {
        return;                                 /* not an injection line */
    }
    char *save = NULL;
    strtok_r(line, ",", &save);                 /* "INJ" */
    char *seq_tok = strtok_r(NULL, ",", &save);
    if (seq_tok == NULL) {
        return;
    }
    uint32_t seq = (uint32_t)strtoul(seq_tok, NULL, 10);

    static int16_t iq[NUM_VALS];
    int n = 0;
    char *tok;
    while (n < NUM_VALS && (tok = strtok_r(NULL, ",", &save)) != NULL) {
        iq[n++] = (int16_t)atoi(tok);
    }
    if (n == NUM_VALS) {
        emit_csi(seq, iq, n);
    }
}

void app_main(void)
{
    /* Install the UART driver on the console port for reliable RX, and route
     * stdio through it so printf() output stays coherent with our reads. */
    uart_driver_install(UART_PORT, RX_BUF_SIZE, 0, 0, NULL, 0);
    uart_vfs_dev_use_driver(UART_PORT);

    printf("csi_inject_serial ready\n");

    /* Assemble lines byte-by-byte (handles both \n and \r line endings). */
    static char line[INJ_LINE_MAX];
    int pos = 0;
    uint8_t ch;
    while (1) {
        int r = uart_read_bytes(UART_PORT, &ch, 1, portMAX_DELAY);
        if (r != 1) {
            continue;
        }
        if (ch == '\n' || ch == '\r') {
            if (pos > 0) {
                line[pos] = '\0';
                handle_line(line);
                pos = 0;
            }
        } else if (pos < INJ_LINE_MAX - 1) {
            line[pos++] = (char)ch;
        }
    }
}
