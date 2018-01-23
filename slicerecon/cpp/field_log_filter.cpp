#include <bulk/bulk.hpp>
#include <bulk/backends/thread/thread.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <eigen3/unsupported/Eigen/FFT>

#define PYBIND11_CPP17
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

void process_projection(int rows, int cols, float *data, float *dark,
                        float *reciproc) {
    // divide work by rows
    auto env = bulk::thread::environment();
    env.spawn(env.available_processors(), [=](auto &world) {
        auto fft = Eigen::FFT<float>();
        auto freq_buffer = std::vector<std::complex<float>>(cols);

        auto s = world.rank();
        auto p = world.active_processors();
        auto block_size = ((rows - 1) / p) + 1;
        auto first_row = s * block_size;
        auto final_row = std::min((s + 1) * block_size, rows);

        auto mid = (cols + 1) / 2;
        for (auto r = first_row; r < final_row; ++r) {
            int index = r * cols;
            for (auto c = 0; c < cols; ++c) {
                data[index] = (data[index] - dark[index]) * reciproc[index];
                data[index] =
                    -std::log(data[index] <= 0.0f ? 1.0f : data[index]);
                index++;
            }

            // filter the row
            fft.fwd(freq_buffer.data(), &data[r * cols], cols);
            // ram-lak filter
            for (int i = 0; i < mid; ++i) {
                freq_buffer[i] *= i;
            }
            for (int j = mid; j < cols; ++j) {
                freq_buffer[j] *= cols - j;
            }
            fft.inv(&data[r * cols], freq_buffer.data(), cols);
        }

        world.sync();
    });
}

PYBIND11_MODULE(slicerecon, m) {
    m.doc() = "c++ implementations for slicerecon";
    m.def("process_projection", &process_projection,
          "Field, log and filter projection");
}
