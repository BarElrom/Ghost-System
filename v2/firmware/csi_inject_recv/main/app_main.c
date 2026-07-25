/*
 * GHOST System v2 — ESP32 CSI Injection Loopback firmware (csi_inject_recv)
 *
 * Option B (firmware loopback). This firmware does NOT sense real RF. It:
 *   1. Joins a Wi-Fi network as a station (SSID/pass via menuconfig).
 *   2. Listens for GHOST "G2" injection frames on a UDP port.
 *   3. For frames addressed to this node, re-emits a standard ESP32 CSI_DATA
 *      CSV line over the UART, byte-compatible with the ghost gateway.
 *
 * Wire frame (big-endian), produced by v2/transport/frame.py encode_frame():
 *   off 0  : magic  "G2"        (2 bytes)
 *   off 2  : version            (1)
 *   off 3  : node_id            (1)   1=RX1 2=RX2 3=RX3
 *   off 4  : flags              (1)   bit0 = calibration (informational here)
 *   off 5  : reserved           (1)
 *   off 6  : frame_seq          (4, uint32 BE)
 *   off 10 : num_sub            (2, uint16 BE)
 *   off 12 : num_sub*2 int16 BE, interleaved I0,Q0,I1,Q1,...
 *
 * The emitted CSI_DATA line matches v2/transport/mock_esp32.py frame_to_csi_line()
 * (only id, timestamp and the I/Q array are faithful; other metadata are fillers).
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_timer.h"

#include "lwip/sockets.h"

static const char *TAG = "csi_inject_recv";

#define GHOST_NODE_ID    CONFIG_GHOST_NODE_ID
#define GHOST_UDP_PORT   CONFIG_GHOST_UDP_PORT

#define WIFI_CONNECTED_BIT BIT0
static EventGroupHandle_t s_wifi_event_group;

/* MAC stamped into the emitted line (matches config_v2.TX_MAC). */
static const uint8_t TX_MAC[6] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};

/* ---------------------------------------------------------------- Wi-Fi STA */
static void event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "disconnected — reconnecting");
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "got IP " IPSTR " — node RX%d ready", IP2STR(&e->ip_info.ip), GHOST_NODE_ID);
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL, NULL));

    wifi_config_t wc = {0};
    strncpy((char *)wc.sta.ssid, CONFIG_GHOST_WIFI_SSID, sizeof(wc.sta.ssid) - 1);
    strncpy((char *)wc.sta.password, CONFIG_GHOST_WIFI_PASSWORD, sizeof(wc.sta.password) - 1);

    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    ESP_LOGI(TAG, "connecting to SSID '%s' ...", CONFIG_GHOST_WIFI_SSID);
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT,
                        pdFALSE, pdTRUE, portMAX_DELAY);
}

/* ---------------------------------------------------------- CSI_DATA output */
static void emit_csi_line(uint32_t seq, const int16_t *iq, int n_vals)
{
    int64_t ts = esp_timer_get_time();  /* microseconds */

    /* 24 metadata fields (indices 0-23), then the quoted CSI array.
     * Fillers match mock_esp32.frame_to_csi_line(); faithful: id, timestamp, I/Q. */
    printf("CSI_DATA,%u," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%d,%d,%d,%d,%d",
           (unsigned)seq, MAC2STR(TX_MAC),
           -45,   /* rssi */
           11,    /* rate */
           1,     /* sig_mode */
           7,     /* mcs */
           1,     /* bandwidth (40 MHz) */
           0,     /* smoothing */
           1,     /* not_sounding */
           0,     /* aggregation */
           0,     /* stbc */
           0,     /* fec_coding */
           0,     /* sgi */
           -95,   /* noise_floor */
           0,     /* ampdu_cnt */
           6,     /* channel */
           0,     /* secondary_channel */
           (long long)ts,  /* local_timestamp */
           0,     /* ant */
           n_vals,/* sig_len */
           1,     /* rx_format */
           n_vals,/* len */
           0);    /* first_word */

    printf(",\"[");
    for (int i = 0; i < n_vals; i++) {
        printf(i ? ",%d" : "%d", iq[i]);
    }
    printf("]\"\n");
}

/* ------------------------------------------------------------ UDP loopback */
static void udp_server_task(void *pv)
{
    uint8_t rx[2048];
    int16_t iq[128];  /* 64 subcarriers * 2 (I,Q) */

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket() failed: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(GHOST_UDP_PORT);
    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        ESP_LOGE(TAG, "bind() failed: errno %d", errno);
        close(sock);
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "UDP loopback listening on :%d for node RX%d",
             GHOST_UDP_PORT, GHOST_NODE_ID);

    while (1) {
        struct sockaddr_in src;
        socklen_t slen = sizeof(src);
        int len = recvfrom(sock, rx, sizeof(rx), 0, (struct sockaddr *)&src, &slen);
        if (len < 12) {
            continue;                       /* too short for a header */
        }
        if (rx[0] != 'G' || rx[1] != '2') {
            continue;                       /* bad magic */
        }
        uint8_t node_id = rx[3];
        uint32_t seq = ((uint32_t)rx[6] << 24) | ((uint32_t)rx[7] << 16) |
                       ((uint32_t)rx[8] << 8) | (uint32_t)rx[9];
        uint16_t num_sub = ((uint16_t)rx[10] << 8) | (uint16_t)rx[11];

        int need = 12 + (int)num_sub * 4;   /* 2 int16 per subcarrier */
        if (len < need) {
            continue;                       /* truncated payload */
        }
        if (node_id != GHOST_NODE_ID) {
            continue;                       /* not addressed to this node */
        }

        int n_vals = (int)num_sub * 2;
        if (n_vals > 128) {
            n_vals = 128;
        }
        for (int k = 0; k < n_vals; k++) {
            int off = 12 + k * 2;
            iq[k] = (int16_t)(((uint16_t)rx[off] << 8) | (uint16_t)rx[off + 1]);
        }
        emit_csi_line(seq, iq, n_vals);
    }
}

/* ----------------------------------------------------------------- app_main */
void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init_sta();

    xTaskCreate(udp_server_task, "udp_server", 4096, NULL, 5, NULL);
}
