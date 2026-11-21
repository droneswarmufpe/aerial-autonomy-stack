#include <iostream>
#include <string>
#include <memory>
#include <csignal>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <gz/transport/Node.hh>
#include <gz/msgs/image.pb.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

// Global objects
GstElement *pipeline = nullptr;
GMainLoop *loop = nullptr;
GstElement *appsrc = nullptr;
bool configured = false;
int configured_width = 0;
int configured_height = 0;
std::string configured_format;

// Command line args
std::string TARGET_IP = "127.0.0.1";
int TARGET_PORT = 5600;
int FRAMERATE = 10;
bool DEPTH_MODE = false;
std::uint64_t output_frame_index = 0;
std::chrono::steady_clock::time_point last_depth_frame_time{};
std::mutex depth_frame_mutex;

bool depth_frame_is_due() {
    std::lock_guard<std::mutex> lock(depth_frame_mutex);
    const auto now = std::chrono::steady_clock::now();
    const auto min_interval = std::chrono::duration<double>(1.0 / FRAMERATE);
    if (last_depth_frame_time.time_since_epoch().count() != 0 &&
        now - last_depth_frame_time < min_interval) {
        return false;
    }
    last_depth_frame_time = now;
    return true;
}

bool depth_image_to_millimeters(const gz::msgs::Image &msg, std::string &output) {
    const std::size_t width = msg.width();
    const std::size_t height = msg.height();
    const std::size_t row_stride = msg.step() ? msg.step() : width * 4;
    const std::string &input = msg.data();

    if (width == 0 || height == 0 || row_stride < width * 2 ||
        input.size() < row_stride * height) {
        return false;
    }

    output.assign(width * height * 2, '\0');
    const std::size_t bytes_per_pixel = row_stride / width;
    if (msg.pixel_format_type() == gz::msgs::L_INT16 && bytes_per_pixel == 2) {
        for (std::size_t y = 0; y < height; ++y) {
            std::memcpy(output.data() + y * width * 2,
                        input.data() + y * row_stride, width * 2);
        }
        return true;
    }

    // Gazebo depth_camera publishes R_FLOAT32 in metres. Convert to the
    // uint16 millimetres expected by RCPilot's GRAY16_LE depth provider.
    if (row_stride < width * sizeof(float)) {
        return false;
    }
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            float meters = 0.0f;
            const char *pixel = input.data() + y * row_stride + x * sizeof(float);
            std::memcpy(&meters, pixel, sizeof(meters));
            std::uint16_t millimeters = 0;
            if (std::isfinite(meters) && meters > 0.0f) {
                const long rounded = std::lround(static_cast<double>(meters) * 1000.0);
                millimeters = static_cast<std::uint16_t>(
                    std::clamp(rounded, 1L, 65535L));
            }
            const std::size_t offset = (y * width + x) * 2;
            output[offset] = static_cast<char>(millimeters & 0xff);
            output[offset + 1] = static_cast<char>((millimeters >> 8) & 0xff);
        }
    }
    return true;
}

std::string detect_gst_raw_format(const gz::msgs::Image &msg) {
    const auto width = static_cast<gsize>(msg.width());
    const auto height = static_cast<gsize>(msg.height());
    const gsize size = msg.data().size();
    if (width == 0 || height == 0 || size == 0) {
        return "";
    }

    const gsize pixels = width * height;
    if (pixels == 0 || size % pixels != 0) {
        return "";
    }

    const gsize bytes_per_pixel = size / pixels;
    switch (msg.pixel_format_type()) {
        case gz::msgs::L_INT8:
            return bytes_per_pixel == 1 ? "GRAY8" : "";
        case gz::msgs::L_INT16:
            if (bytes_per_pixel != 2) return "";
#if G_BYTE_ORDER == G_LITTLE_ENDIAN
            return "GRAY16_LE";
#else
            return "GRAY16_BE";
#endif
        case gz::msgs::R_FLOAT32:
            return bytes_per_pixel == 4 ? "RGBA" : "";
        case gz::msgs::RGB_INT8:
            return bytes_per_pixel == 3 ? "RGB" : "";
        case gz::msgs::RGBA_INT8:
            return bytes_per_pixel == 4 ? "RGBA" : "";
        case gz::msgs::BGR_INT8:
            return bytes_per_pixel == 3 ? "BGR" : "";
        case gz::msgs::BGRA_INT8:
            return bytes_per_pixel == 4 ? "BGRA" : "";
        default:
            // Older publishers may not populate pixel_format_type.
            switch (bytes_per_pixel) {
                case 1:
                    return "GRAY8";
                case 2:
#if G_BYTE_ORDER == G_LITTLE_ENDIAN
                    return "GRAY16_LE";
#else
                    return "GRAY16_BE";
#endif
                case 3:
                    return "RGB";
                case 4:
                    return "RGBA";
                default:
                    return "";
            }
    }
}

void configure_caps_if_needed(const gz::msgs::Image &msg, const std::string &format) {
    const bool changed = !configured ||
        configured_width != static_cast<int>(msg.width()) ||
        configured_height != static_cast<int>(msg.height()) ||
        configured_format != format;

    if (!changed) {
        return;
    }

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, format.c_str(),
        "width", G_TYPE_INT, msg.width(),
        "height", G_TYPE_INT, msg.height(),
        "framerate", GST_TYPE_FRACTION, FRAMERATE, 1,
        NULL);

    gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
    gst_caps_unref(caps);

    configured = true;
    configured_width = static_cast<int>(msg.width());
    configured_height = static_cast<int>(msg.height());
    configured_format = format;

    std::cout << "Configured GStreamer for " << msg.width() << "x" << msg.height()
              << " format=" << format << std::endl;
}

// Image callback
void on_frame(const gz::msgs::Image &msg) {
    if (!appsrc) return;
    if (DEPTH_MODE && !depth_frame_is_due()) return;

    std::string format;
    std::string payload;
    if (DEPTH_MODE) {
        format = "GRAY16_LE";
        if (!depth_image_to_millimeters(msg, payload)) {
            g_warning("Unsupported depth image layout: width=%u height=%u step=%u bytes=%zu",
                      msg.width(), msg.height(), msg.step(), msg.data().size());
            return;
        }
    } else {
        format = detect_gst_raw_format(msg);
        if (format.empty()) {
            g_warning("Unsupported image layout: width=%u height=%u bytes=%zu",
                      msg.width(), msg.height(), msg.data().size());
            return;
        }
        payload = msg.data();
    }
    configure_caps_if_needed(msg, format);

    const gsize size = payload.size();
    GstBuffer *buffer = gst_buffer_new_allocate(NULL, size, NULL);
    if (!buffer) {
        g_warning("Failed to allocate GStreamer buffer");
        return;
    }

    gst_buffer_fill(buffer, 0, payload.data(), size);
    if (DEPTH_MODE) {
        const GstClockTime duration = gst_util_uint64_scale(1, GST_SECOND, FRAMERATE);
        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(output_frame_index, GST_SECOND, FRAMERATE);
        GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
        GST_BUFFER_DURATION(buffer) = duration;
        ++output_frame_index;
    }

    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
    if (ret != GST_FLOW_OK) {
        g_warning("Error pushing buffer to appsrc");
    }
}

bool check_nvidia_encoder() {
    GstElementFactory *factory = gst_element_factory_find("nvh264enc");
    if (factory) {
        gst_object_unref(factory);
        return true;
    }
    return false;
}

void sigint_handler(int sig) {
    (void)sig; // Suppress unused warning
    if (loop) g_main_loop_quit(loop);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./gz_gst_bridge <gz_topic> <ip> [port] [framerate] [color|depth]\n";
        return 1;
    }
    std::string topic = argv[1];
    TARGET_IP = argv[2];
    if (argc > 3) TARGET_PORT = std::stoi(argv[3]);
    if (argc > 4) FRAMERATE = std::stoi(argv[4]);
    const std::string stream_mode = argc > 5 ? argv[5] : "color";
    if (stream_mode != "color" && stream_mode != "depth") {
        std::cerr << "Stream mode must be color or depth\n";
        return 1;
    }
    DEPTH_MODE = stream_mode == "depth";
    if (FRAMERATE <= 0) {
        std::cerr << "Framerate must be positive\n";
        return 1;
    }

    // Initialize GStreamer and Gazebo Transport
    gst_init(&argc, &argv);    
    gz::transport::Node node;

    // Build Pipeline String
    std::string pipeline_str;
    std::string ip_port = "host=" + TARGET_IP + " port=" + std::to_string(TARGET_PORT);
    if (DEPTH_MODE) {
        std::cout << "Using lossless GRAY16_LE depth payload (rtpgstpay)\n";
        pipeline_str = "appsrc name=gz_source ! queue max-size-buffers=1 leaky=downstream ! "
                       "rtpgstpay pt=96 mtu=1400 config-interval=1 ! udpsink sync=false " + ip_port;
    } else if (check_nvidia_encoder()) {
        std::cout << "Using NVIDIA GPU Encoder (nvh264enc)\n";
        pipeline_str = "appsrc name=gz_source ! queue max-size-buffers=1 leaky=downstream ! "
                       "videoconvert ! "
                       "nvh264enc preset=low-latency-hq zerolatency=true rc-mode=cbr bitrate=2048 qp-min=15 qp-max=35 gop-size=60 ! "
                       "rtph264pay config-interval=1 mtu=1400 ! udpsink sync=false " + ip_port;
    } else {
        std::cout << "Using CPU Encoder (x264enc)\n";
        pipeline_str = "appsrc name=gz_source ! queue max-size-buffers=1 leaky=downstream ! "
                       "videoconvert ! "
                       "x264enc speed-preset=ultrafast tune=zerolatency bitrate=2048 key-int-max=60 ! "
                       "rtph264pay config-interval=1 mtu=1400 ! udpsink sync=false " + ip_port;
    }
    // Note: re-send full frame every 60 frames

    // Launch Pipeline
    GError *error = nullptr;
    pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline) {
        std::cerr << "Failed to create pipeline: " << error->message << std::endl;
        return -1;
    }

    appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "gz_source");
    g_object_set(appsrc, "format", GST_FORMAT_TIME, NULL);
    g_object_set(appsrc, "is-live", TRUE, NULL);
    g_object_set(appsrc, "leaky-type", 2, NULL); // Drop old frames
    g_object_set(appsrc, "max-bytes", 0, NULL);
    g_object_set(appsrc, "max-buffers", 2, NULL); 
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    if (!node.Subscribe(topic, on_frame)) {
        std::cerr << "Error subscribing to " << topic << std::endl;
        return -1;
    }
    
    std::cout << "Streaming " << topic << " to " << TARGET_IP << ":" << TARGET_PORT << " at " << FRAMERATE << " FPS." << std::endl;

    loop = g_main_loop_new(NULL, FALSE);
    signal(SIGINT, sigint_handler);
    g_main_loop_run(loop);

    // Cleanup
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsrc);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);

    return 0;
}
