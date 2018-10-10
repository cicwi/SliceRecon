#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

#include "tomop/tomop.hpp"
#include <zmq.hpp>

#include "../util/data_types.hpp"

namespace slicerecon {

class visualization_server {
  public:
    using callback_type =
        std::function<slice_data(std::array<float, 9>, int32_t)>;

    visualization_server(
        std::string name, std::string hostname = "tcp://localhost:5555",
        std::string subscribe_hostname = "tcp://localhost:5556")
        : context_(1), socket_(context_, ZMQ_REQ),
          subscribe_socket_(context_, ZMQ_SUB) {
        using namespace std::chrono_literals;

        // set socket timeout to 200 ms
        socket_.setsockopt(ZMQ_LINGER, 200);
        socket_.connect(hostname);

        auto packet = tomop::MakeScenePacket(name, 3);

        packet.send(socket_);

        // check if we get a reply within a second
        zmq::pollitem_t items[] = {{socket_, 0, ZMQ_POLLIN, 0}};
        auto poll_result = zmq::poll(items, 1, 1000ms);

        if (poll_result <= 0) {
            throw tomop::server_error("Could not connect to server");
        } else {
            //  get the reply.
            zmq::message_t reply;
            socket_.recv(&reply);
            scene_id_ = *(int32_t*)reply.data();
        }

        subscribe(subscribe_hostname);

        util::log << LOG_FILE << util::lvl::info
                  << "Connected to visualization server: " << hostname << " "
                  << subscribe_hostname << util::end_log;
    }

    ~visualization_server() {
        socket_.close();
        subscribe_socket_.close();
        context_.close();

        if (serve_thread_.joinable()) {
            serve_thread_.join();
        }
    }

    void send(const tomop::Packet& packet) {
        packet.send(socket_);
        zmq::message_t reply;
        socket_.recv(&reply);
    }

    void subscribe(std::string subscribe_host) {
        if (scene_id_ < 0) {
            throw tomop::server_error(
                "Subscribe called for uninitialized server");
        }

        // set socket timeout to 200 ms
        socket_.setsockopt(ZMQ_LINGER, 200);

        //  Socket to talk to server
        subscribe_socket_.connect(subscribe_host);

        std::vector<tomop::packet_desc> descriptors = {
            tomop::packet_desc::set_slice, tomop::packet_desc::remove_slice,
            tomop::packet_desc::kill_scene};

        for (auto descriptor : descriptors) {
            int32_t filter[] = {
                (std::underlying_type<tomop::packet_desc>::type)descriptor,
                scene_id_};
            subscribe_socket_.setsockopt(ZMQ_SUBSCRIBE, filter,
                                         sizeof(decltype(filter)));
        }
    }

    void serve() {
        // not only serve this, but also other..
        auto recast_thread = std::thread([&] {
            while (true) {
                zmq::message_t update;
                bool kill = false;
                if (!subscribe_socket_.recv(&update)) {
                    kill = true;
                } else {
                    auto desc = ((tomop::packet_desc*)update.data())[0];
                    auto buffer = tomop::memory_buffer(update.size(),
                                                       (char*)update.data());

                    switch (desc) {
                    case tomop::packet_desc::kill_scene: {
                        auto packet =
                            std::make_unique<tomop::KillScenePacket>();
                        packet->deserialize(std::move(buffer));

                        if (packet->scene_id != scene_id_) {
                            std::cout << "Received kill request with wrong "
                                         "scene id\n";
                        } else {
                            kill = true;
                        }
                        break;
                    }

                    case tomop::packet_desc::set_slice: {
                        auto packet = std::make_unique<tomop::SetSlicePacket>();
                        packet->deserialize(std::move(buffer));

                        make_slice(packet->slice_id, packet->orientation);
                        break;
                    }
                    case tomop::packet_desc::remove_slice: {
                        auto packet =
                            std::make_unique<tomop::RemoveSlicePacket>();
                        packet->deserialize(std::move(buffer));

                        auto to_erase = std::find_if(
                            slices_.begin(), slices_.end(), [&](auto x) {
                                return x.first == packet->slice_id;
                            });
                        slices_.erase(to_erase);
                        break;
                    }
                    default:
                        break;
                    }
                }

                if (kill) {
                    std::cout << "Scene closed...\n";
                    break;
                }
            }
        });

        recast_thread.join();
    }

    void make_slice(int32_t slice_id, std::array<float, 9> orientation) {
        int32_t update_slice_index = -1;
        int32_t i = 0;
        for (auto& id_and_slice : slices_) {
            if (id_and_slice.first == slice_id) {
                update_slice_index = i;
                break;
            }
            ++i;
        }

        if (update_slice_index >= 0) {
            slices_[update_slice_index] = std::make_pair(slice_id, orientation);
        } else {
            slices_.push_back(std::make_pair(slice_id, orientation));
        }

        if (!slice_data_callback_) {
            throw tomop::server_error("No callback set");
        }

        auto result = slice_data_callback_(orientation, slice_id);

        if (!result.first.empty()) {
            auto data_packet =
                tomop::SliceDataPacket(scene_id_, slice_id, result.first, false,
                                       std::move(result.second));
            send(data_packet);
        }
    }

    void set_slice_callback(callback_type callback) {
        slice_data_callback_ = callback;
    }

    int32_t scene_id() { return scene_id_; }

  private:
    // server connection
    zmq::context_t context_;
    zmq::socket_t socket_;

    // subscribe connection
    std::thread serve_thread_;
    zmq::socket_t subscribe_socket_;

    int32_t scene_id_ = -1;

    callback_type slice_data_callback_;
    std::vector<std::pair<int32_t, std::array<float, 9>>> slices_;
};

} // namespace slicerecon