// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS("src/python/index.cpp", line)

#include <pybind11/numpy.h>
#include <awkward/python/content.h>

#include "awkward/python/util.h"

#include "awkward/python/index.h"

#include "awkward/python/dlpack_util.h"

template <typename T>
const ak::IndexOf<T>
Index_from_cuda_array_interface(const std::string& name,
                                const py::object& array) {
  py::dict cuda_array_interface = array.attr("__cuda_array_interface__");

  const std::vector<ssize_t> shape = cuda_array_interface["shape"].cast<std::vector<ssize_t>>();
  std::string typestr = cuda_array_interface["typestr"].cast<std::string>();

  if(shape.empty()) {
    throw std::invalid_argument(
        std::string("Array must not be scalar; try array.reshape(1)")
        + FILENAME(__LINE__));
  }

  if (shape.size() != 1) {
    throw std::invalid_argument(
      name + std::string(" must be built from a one-dimensional array; "
                         "try array.ravel()") + FILENAME(__LINE__));
  }

  const uint8_t dtype_size = std::stoi(typestr.substr(2));
  const char dtype_code = typestr[1];
  ak::util::dtype array_dtype;

  if (typestr.length() >= 3) {
    int32_t test = 1;
    bool little_endian = (*(int8_t*)&test == 1);
    std::string endianness = typestr.substr(0, 1);
    if ((endianness == ">"  &&  !little_endian)  ||
        (endianness == "<"  &&  little_endian)  ||
        (endianness == "=")) {

      switch(dtype_code) {
        case 'b': array_dtype = ak::util::dtype::boolean;
                  break;
        case 'i': if (dtype_size == 1) {
                    array_dtype = ak::util::dtype::int8;
                  }
                  else if (dtype_size == 2) {
                    array_dtype = ak::util::dtype::int16;
                  }
                  else if (dtype_size == 4) {
                    array_dtype = ak::util::dtype::int32;
                  }
                  else if (dtype_size == 8) {
                    array_dtype = ak::util::dtype::int64;
                  }
                  break;
        case 'u': if (dtype_size == 1) {
                    array_dtype = ak::util::dtype::uint8;
                  }
                  else if (dtype_size == 2) {
                    array_dtype = ak::util::dtype::uint16;
                  }
                  else if (dtype_size == 4) {
                    array_dtype = ak::util::dtype::uint32;
                  }
                  else if (dtype_size == 8) {
                    array_dtype = ak::util::dtype::uint64;
                  }
                  break;

        case 'f': if (dtype_size == 2) {
                    array_dtype = ak::util::dtype::float16;
                  }
                  else if (dtype_size == 4) {
                    array_dtype = ak::util::dtype::float32;
                  }
                  else if (dtype_size == 8) {
                    array_dtype = ak::util::dtype::float64;
                  }
                  else if (dtype_size == 16) {
                    array_dtype = ak::util::dtype::float128;
                  }
                  break;

        case 'c': if (dtype_size == 8) {
                    array_dtype = ak::util::dtype::complex64;
                  }
                  else if (dtype_size == 16) {
                    array_dtype = ak::util::dtype::complex128;
                  }
                  else if (dtype_size == 32) {
                    array_dtype = ak::util::dtype::complex256;
                  }
                  break;

        default: std::invalid_argument(std::string("Couldn't find a compatible ak::dtype for given typestr: ") + typestr + FILENAME(__LINE__));
      }
    }
    else if ((endianness == ">"  &&  little_endian)  ||
             (endianness == "<"  &&  !little_endian)) {
      throw std::invalid_argument(std::string("Input Array has a different endianness than the System") + FILENAME(__LINE__));
    }
  }

  if (array_dtype != ak::util::name_to_dtype(py::cast<std::string>(py::str(py::dtype::of<T>())))) {
    throw std::invalid_argument(
      name + std::string(" arg0: must be a ")
      + py::cast<std::string>(py::str(py::dtype::of<T>()))
      + std::string(" array") + FILENAME(__LINE__));
  }

  std::vector<ssize_t> form_strides;
  if(cuda_array_interface.contains("strides") && !cuda_array_interface["strides"].is_none()) {
    form_strides = cuda_array_interface["strides"].cast<std::vector<ssize_t>>();
  }
  else {
    form_strides = cuda_array_interface["shape"].cast<std::vector<ssize_t>>();
    form_strides[0] = 1;
    std::transform(form_strides.begin(), form_strides.end(), form_strides.begin(),
      [dtype_size](ssize_t& form_strides_ele) -> ssize_t { return form_strides_ele * dtype_size; });
    std::reverse(form_strides.begin(), form_strides.end());
  }
  const std::vector<ssize_t> strides = form_strides;

  if (strides[0] != sizeof(T)) {
    throw std::invalid_argument(
      name + std::string(" must be built from a contiguous array "
                         "(array.strides == (array.itemsize,)); "
                         "try array.copy()") + FILENAME(__LINE__));
  }

  T* ptr = reinterpret_cast<T*>(cuda_array_interface["data"].cast<std::vector<ssize_t>>()[0]);

  return ak::IndexOf<T>(std::shared_ptr<T>(ptr, pyobject_deleter<T>(array.ptr())),
                        0,
                        (int64_t)shape[0],
                        ak::kernel::lib::cuda);
}

template <typename T>
ak::IndexOf<T>
Index_from_cupy(const std::string& name, const py::object& array) {
  if(py::hasattr(array, "__cuda_array_interface__")) {
    return Index_from_cuda_array_interface<T>(name, array);
  }
  if (py::isinstance(array, py::module::import("cupy").attr("ndarray"))) {
    if (!py::dtype(array.attr("dtype")).equal(py::dtype::of<T>())) {
      throw std::invalid_argument(
        name + std::string(" arg0: must be a ")
        + py::cast<std::string>(py::str(py::dtype::of<T>()))
        + std::string(" array") + FILENAME(__LINE__));
    }

    if (py::cast<int64_t>(array.attr("ndim")) != 1) {
      throw std::invalid_argument(
        name + std::string(" must be built from a one-dimensional array; "
                           "try array.ravel()") + FILENAME(__LINE__));
    }

    std::vector<int64_t> strides = array.attr("strides").cast<std::vector<int64_t>>();

    if (strides[0] != sizeof(T)) {
      throw std::invalid_argument(
        name + std::string(" must be built from a contiguous array "
                           "(array.strides == (array.itemsize,)); "
                           "try array.copy()") + FILENAME(__LINE__));
    }

    T* ptr = reinterpret_cast<T*>(py::cast<ssize_t>(array.attr("data").attr("ptr")));

    std::vector<int64_t> shape = array.attr("shape").cast<std::vector<int64_t>>();

    return ak::IndexOf<T>(std::shared_ptr<T>(ptr, pyobject_deleter<T>(array.ptr())),
                          0,
                          (int64_t)shape[0],
                          ak::kernel::lib::cuda);
  }
  else {
    throw std::invalid_argument(
      name + std::string(".from_cupy() can only accept CuPy Arrays!")
      + FILENAME(__LINE__));
  }
}

template <typename T>
ak::IndexOf<T>
Index_from_jax(const std::string& name, const py::object& array) {
  const std::string device = array.attr("device_buffer").attr("device")().attr("platform").cast<std::string>();

  if (device.compare("cpu") == 0) {
    py::array_t<T> jax_array = array.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();

    py::buffer_info info = jax_array.request();
    if (info.ndim != 1) {
      throw std::invalid_argument(
        name + std::string(" must be built from a one-dimensional array; "
                           "try array.ravel()") + FILENAME(__LINE__));
    }
    if (info.strides[0] != sizeof(T)) {
      throw std::invalid_argument(
        name + std::string(" must be built from a contiguous array "
                           "(array.strides == (array.itemsize,)); "
                           "try array.copy()") + FILENAME(__LINE__));
    }
    return ak::IndexOf<T>(
      std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr),
                         pyobject_deleter<T>(array.ptr())),
      0,
      (int64_t)info.shape[0],
      ak::kernel::lib::cpu);
  }
  else if (device.compare("gpu") == 0) {
    if(py::hasattr(array, "__cuda_array_interface__")) {
      return Index_from_cuda_array_interface<T>(name, array);
    }
    else {
      throw std::invalid_argument(
        name + std::string(".from_jaxgpu() needs a __cuda_array_interface__ dict of the given array, to accept JAX GPU buffers")
        + FILENAME(__LINE__));
    }
  }
  else {
    throw std::invalid_argument(
        std::string("Awkward Arrays don't support ") + device + FILENAME(__LINE__));
  }
}

template <typename T>
py::class_<ak::IndexOf<T>>
make_IndexOf(const py::handle& m, const std::string& name) {
  return (py::class_<ak::IndexOf<T>>(m, name.c_str(), py::buffer_protocol())
      .def_buffer([](const ak::IndexOf<T>& self) -> py::buffer_info {
        return py::buffer_info(
          reinterpret_cast<void*>(reinterpret_cast<ssize_t>(self.ptr().get())
                                  + self.offset()*sizeof(T)),
          sizeof(T),
          py::format_descriptor<T>::format(),
          1,
          { (ssize_t)self.length() },
          { (ssize_t)sizeof(T) });
        })

      .def(py::init([name](const py::object& anyarray) -> ak::IndexOf<T> {
        std::string module = anyarray.get_type().attr("__module__").cast<std::string>();
        if (module.rfind("cupy.", 0) == 0) {
          return Index_from_cupy<T>(name, anyarray);
        }
        else if(module.rfind("jax.", 0) == 0) {
          return Index_from_jax<T>(name, anyarray);
        }

        py::array_t<T> array = anyarray.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();

        py::buffer_info info = array.request();
        if (info.ndim != 1) {
          throw std::invalid_argument(
            name + std::string(" must be built from a one-dimensional array; "
                               "try array.ravel()") + FILENAME(__LINE__));
        }
        if (info.strides[0] != sizeof(T)) {
          throw std::invalid_argument(
            name + std::string(" must be built from a contiguous array "
                               "(array.strides == (array.itemsize,)); "
                               "try array.copy()") + FILENAME(__LINE__));
        }
        return ak::IndexOf<T>(
          std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr),
                             pyobject_deleter<T>(array.ptr())),
          0,
          (int64_t)info.shape[0],
          ak::kernel::lib::cpu);
      }))

      .def_property_readonly("ptr_lib", [](const ak::IndexOf<T>& self) {
        if (self.ptr_lib() == ak::kernel::lib::cpu) {
          return py::cast("cpu");
        }
        else if (self.ptr_lib() == ak::kernel::lib::cuda) {
          return py::cast("cuda");
        }
        else {
          throw std::runtime_error(
            std::string("unrecognized ptr_lib") + FILENAME(__LINE__));
        }
      })
      .def("__repr__", &ak::IndexOf<T>::tostring)
      .def("__len__", &ak::IndexOf<T>::length)
      .def("__getitem__", [](const ak::IndexOf<T>& self, py::object& obj) {
        if (py::isinstance<py::int_>(obj)) {
          T out = self.getitem_at(obj.cast<int64_t>());
          return py::cast(out);
        }
        else if (py::isinstance<py::slice>(obj)) {
          py::object pystep = obj.attr("step");
          if ((py::isinstance<py::int_>(pystep)  &&  pystep.cast<int64_t>() == 1)  ||
              pystep.is(py::none())) {
            int64_t start = ak::Slice::none();
            int64_t stop = ak::Slice::none();
            py::object pystart = obj.attr("start");
            py::object pystop = obj.attr("stop");
            if (!pystart.is(py::none())) {
              start = pystart.cast<int64_t>();
            }
            if (!pystop.is(py::none())) {
              stop = pystop.cast<int64_t>();
            }
            ak::IndexOf<T> out = self.getitem_range(start, stop);
            return py::cast(out);
          }
          else {
            throw std::invalid_argument(
              std::string("Index slices cannot contain step != 1")
              + FILENAME(__LINE__));
          }
        }
        else {
          throw std::invalid_argument(
            std::string("Index can only be sliced by an integer or start:stop slice")
            + FILENAME(__LINE__));
        }
      })
      .def_static("from_cupy", [name](const py::object& array) -> py::object {
        return py::cast(Index_from_cupy<T>(name, array));
      })
      .def_static("from_jax", [name](const py::object& array) -> py::object {
        return py::cast(Index_from_jax<T>(name, array));
      })
      .def("copy_to", [name](const ak::IndexOf<T>& self, std::string& ptr_lib) -> py::object {
        if (ptr_lib == "cuda") {
          auto cuda_index = self.copy_to(ak::kernel::lib::cuda);
          return py::cast(cuda_index);
        }
        else if (ptr_lib == "cpu") {
          ak::IndexOf<T> cuda_index = self.copy_to(ak::kernel::lib::cpu) ;
          return py::cast(cuda_index);
        }
        else {
          throw std::invalid_argument(
            std::string("specify 'cpu' or 'cuda'") + FILENAME(__LINE__));
        }
      })
      .def("to_cupy", [name](const ak::IndexOf<T>& self) -> py::object {
        if (self.ptr_lib() != ak::kernel::lib::cuda) {
          throw std::invalid_argument(
            name
            + std::string(" resides in main memory, must be converted to NumPy, not CuPy")
            + FILENAME(__LINE__));
        }

        py::object cupy_unowned_mem = py::module::import("cupy").attr("cuda").attr("UnownedMemory")(
            reinterpret_cast<ssize_t>(self.ptr().get()),
            self.length() * sizeof(T),
            self);

        py::object cupy_memoryptr = py::module::import("cupy").attr("cuda").attr("MemoryPointer")(
            cupy_unowned_mem,
            0);
        py::object cuda_array = py::module::import("cupy").attr("ndarray")(
            pybind11::make_tuple(py::cast<ssize_t>(self.length())),
            py::format_descriptor<T>::format(),
            cupy_memoryptr,
            pybind11::make_tuple(py::cast<ssize_t>(sizeof(T))));

        return py::module::import("cupy").attr("ndarray")(
                pybind11::make_tuple(py::cast<ssize_t>(self.length())),
                py::format_descriptor<T>::format(),
                cupy_memoryptr,
                pybind11::make_tuple(py::cast<ssize_t>(sizeof(T))));
        })
      .def("to_jax", [name](const ak::IndexOf<T>& self) -> py::object {
        DLManagedTensor* dlm_tensor = new DLManagedTensor;

        dlm_tensor->dl_tensor.data = reinterpret_cast<void*>(self.ptr().get());
        dlm_tensor->dl_tensor.ndim = 1;
        dlm_tensor->dl_tensor.dtype = ak::dlpack::data_type_dispatch(ak::util::name_to_dtype(
          py::cast<std::string>(py::str(py::dtype::of<T>()))));

        int64_t* dup_shape = new int64_t[1];
        int64_t* dup_strides = new int64_t[1];

        dup_shape[0] = self.length();
        dup_strides[0] = 1;

        dlm_tensor->dl_tensor.shape = dup_shape;
        dlm_tensor->dl_tensor.strides = dup_strides;
        dlm_tensor->dl_tensor.byte_offset = 0;
        dlm_tensor->dl_tensor.ctx = ak::dlpack::device_context_dispatch(self.ptr_lib(), self.ptr().get());

        py::object array = py::cast(self);
        dlm_tensor->manager_ctx = reinterpret_cast<void*>(array.ptr());

        Py_INCREF(array.ptr());
        dlm_tensor->deleter = ak::dlpack::dlpack_deleter;

        return py::module::import("jax.dlpack").attr("from_dlpack")
                        (py::capsule(dlm_tensor, "dltensor", ak::dlpack::pycapsule_deleter));
      })
  );
}

template py::class_<ak::Index8>
make_IndexOf<int8_t>(const py::handle& m, const std::string& name);

template py::class_<ak::IndexU8>
make_IndexOf<uint8_t>(const py::handle& m, const std::string& name);

template py::class_<ak::Index32>
make_IndexOf<int32_t>(const py::handle& m, const std::string& name);

template py::class_<ak::IndexU32>
make_IndexOf<uint32_t>(const py::handle& m, const std::string& name);

template py::class_<ak::Index64>
make_IndexOf<int64_t>(const py::handle& m, const std::string& name);
