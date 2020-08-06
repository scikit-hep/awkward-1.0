// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "awkward/python/util.h"

#include "awkward/python/identities.h"

template <typename T>
py::class_<ak::IdentitiesOf<T>>
make_IdentitiesOf(const py::handle& m, const std::string& name) {
  return (py::class_<ak::IdentitiesOf<T>>(m,
                                          name.c_str(),
                                          py::buffer_protocol())
      .def_buffer([](const ak::IdentitiesOf<T>& self) -> py::buffer_info {
        return py::buffer_info(
          reinterpret_cast<void*>(
            reinterpret_cast<ssize_t>(
              self.ptr().get()) + self.offset()*sizeof(T)),
          sizeof(T),
          py::format_descriptor<T>::format(),
          2,
          { (ssize_t)self.length(), (ssize_t)self.width() },
          { (ssize_t)(sizeof(T)*self.width()), (ssize_t)sizeof(T) });
        })

      .def_static("newref", &ak::Identities::newref)

      .def(py::init([](ak::Identities::Ref ref,
                       const ak::Identities::FieldLoc& fieldloc,
                       int64_t width,
                       int64_t length) {
        return ak::IdentitiesOf<T>(ref, fieldloc, width, length);
      }))

      .def(py::init([name](ak::Identities::Ref ref,
                           ak::Identities::FieldLoc fieldloc,
                           py::array_t<T,
                           py::array::c_style | py::array::forcecast> array) {
        py::buffer_info info = array.request();
        if (info.ndim != 2) {
          throw std::invalid_argument(
            name + std::string(" must be built from a two-dimensional array"));
        }
        if (info.strides[0] != sizeof(T)*info.shape[1]  ||
            info.strides[1] != sizeof(T)) {
          throw std::invalid_argument(
            name + std::string(" must be built from a contiguous array (array"
                               ".stries == (array.shape[1]*array.itemsize, "
                               "array.itemsize)); try array.copy()"));
        }
        return ak::IdentitiesOf<T>(ref,
                                   fieldloc,
                                   0,
                                   info.shape[1],
                                   info.shape[0],
            std::shared_ptr<T>(reinterpret_cast<T*>(info.ptr),
                               pyobject_deleter<T>(array.ptr())));
      }))

      .def("__repr__", &ak::IdentitiesOf<T>::tostring)
      .def("__len__", &ak::IdentitiesOf<T>::length)
      .def("__getitem__", &ak::IdentitiesOf<T>::getitem_at)
      .def("__getitem__", &ak::IdentitiesOf<T>::getitem_range)

      .def_property_readonly("ref", &ak::IdentitiesOf<T>::ref)
      .def_property_readonly("fieldloc", &ak::IdentitiesOf<T>::fieldloc)
      .def_property_readonly("width", &ak::IdentitiesOf<T>::width)
      .def_property_readonly("length", &ak::IdentitiesOf<T>::length)
      .def_property_readonly("array", [](const py::buffer& self) -> py::array {
        return py::array(self);
      })
      .def("identity_at_str", &ak::IdentitiesOf<T>::identity_at)
      .def("identity_at",
           [](const ak::IdentitiesOf<T>& self, int64_t at) -> py::tuple {
        ak::Identities::FieldLoc fieldloc = self.fieldloc();
        py::tuple out((size_t)self.width() + fieldloc.size());
        size_t j = 0;
        for (int64_t i = 0;  i < self.width();  i++) {
          out[j] = py::cast(self.value(at, i));
          j++;
          for (auto pair : fieldloc) {
            if (pair.first == i) {
              out[j] = py::cast(pair.second);
              j++;
            }
          }
        }
        return out;
      })

      .def_static("from_cupy", [name](ak::Identities::Ref ref,
                                const ak::Identities::FieldLoc& fieldloc,
                                py::object array) {
        if(py::isinstance(array, py::module::import("cupy").attr("ndarray"))) {

          void* ptr = reinterpret_cast<void *>(py::cast<ssize_t>
            (array.attr("data").attr("ptr")));

          if (py::cast<int64_t>(array.attr("ndim")) != 2) {
            throw std::invalid_argument(
              name + std::string(" must be built from a two-dimensional array"));
          }
          std::vector<int64_t> shape, strides;

          shape = pytuples_to_vector<int64_t>(array.attr("shape"));
          strides = pytuples_to_vector<int64_t>(array.attr("strides"));

          if (strides[0] != sizeof(T)*shape[1]  ||
              strides[1] != sizeof(T)) {
            throw std::invalid_argument(
              name + std::string(" must be built from a contiguous array (array"
                                 ".stries == (array.shape[1]*array.itemsize, "
                                 "array.itemsize)); try array.copy()"));
          }
          return ak::IdentitiesOf<T>(ref,
                                     fieldloc,
                                     0,
                                     shape[1],
                                     shape[0],
                                     std::shared_ptr<T>(reinterpret_cast<T*>(ptr),
                                           pyobject_deleter<T>(array.ptr())));
        }
        else {
          throw std::invalid_argument(name + std::string(
            ".from_cupy() can only accept CuPy Arrays!"));
        }
    })
    .def("copy_to",
         [name](const ak::IdentitiesOf<T>& self, const std::string& ptr_lib) -> py::object {
             if (ptr_lib == "cpu") {
               auto cpu_identities = self.copy_to(ak::kernel::lib::cpu) ;
               return py::cast<ak::IdentitiesOf<T>>(*cpu_identities);
             }
             else if (ptr_lib == "cuda") {
               auto cuda_identities = self.copy_to(ak::kernel::lib::cuda);

               auto cupy_unowned_mem = py::module::import("cupy").attr("cuda").attr("UnownedMemory")(
                 reinterpret_cast<ssize_t>(cuda_identities->ptr().get()),
                 cuda_identities->length() * sizeof(T),
                 cuda_identities);

               auto cupy_memoryptr = py::module::import("cupy").attr("cuda").attr("MemoryPointer")(
                 cupy_unowned_mem,
                 0);

               return py::module::import("awkward1").attr("layout").attr(name.c_str()).attr("from_cupy")
                 (py::module::import("cupy").attr("ndarray")(
                   pybind11::make_tuple(py::cast<ssize_t>(cuda_identities->length())),
                   py::format_descriptor<T>::format(),
                   cupy_memoryptr,
                   pybind11::make_tuple(py::cast<ssize_t>(sizeof(T)))));
             }
             else {
               throw std::invalid_argument("specify 'cpu' or 'cuda'");
             }
         })

  );
}

template py::class_<ak::Identities32>
make_IdentitiesOf(const py::handle& m, const std::string& name);

template py::class_<ak::Identities64>
make_IdentitiesOf(const py::handle& m, const std::string& name);
