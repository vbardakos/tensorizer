use pyo3::{PyResult, ffi, prelude::pyclass, pymethods};
use std::{ffi::c_int, mem, ptr};

// note: Python Allocator is fed raw bytes & already knows dtype
#[pyclass(frozen)]
pub(crate) struct SimdResult(Vec<u8>);

impl<T> From<Vec<T>> for SimdResult
where
    T: Copy,
{
    fn from(data: Vec<T>) -> Self {
        let mut data = mem::ManuallyDrop::new(data); // copium kills
        let transmuted = unsafe {
            Vec::from_raw_parts(
                data.as_mut_ptr() as *mut u8,
                mem::size_of_val(data.as_slice()),
                data.capacity() * mem::size_of::<T>(),
            )
        };

        Self(transmuted)
    }
}

/*
 * note :: PyO3 maps directly to CPython's buffer protocol slots
 * source: https://peps.python.org/pep-0688/#python-level-buffer-protocol
 */
#[pymethods]
impl SimdResult {
    /*
     * note :: Instructs Python where data live and how to read it.
     * assignments validity source: https://docs.python.org/3/c-api/buffer.html
     */
    #[rustfmt::skip]
    unsafe fn __getbuffer__(&self, view: *mut ffi::Py_buffer, _flag: c_int) -> PyResult<()> {
        unsafe {
            (*view).buf = self.0.as_ptr() as *mut _;   // ptr to raw bytes
            (*view).len = self.0.len() as isize;       // byte count
            (*view).readonly = 1;                      // immutable
            (*view).itemsize = 1;                      // 1 byte per item
            (*view).ndim = 1;                          // flat
            (*view).format = ptr::null_mut();          // no format — consumer knows the dtype
            (*view).shape = ptr::null_mut();           // null is valid for contiguous 1D buffers
            (*view).strides = ptr::null_mut();         // null means tightly packed (stride = 1)
            (*view).suboffsets = ptr::null_mut();      // no indirect dimensions
            (*view).internal = ptr::null_mut();        // unused by CPython
        }

        Ok(())
    }

    // note :: Nothing to defer. Vec is owned by rust and released during gc
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}
