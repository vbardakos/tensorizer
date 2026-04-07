#![feature(portable_simd)]

mod ops;
mod result;

use pyo3::prelude::{pymodule, Python};
use pyo3::buffer::{Element, PyBuffer, ReadOnlyCell};


pub(crate) trait IntoSlice<'a, T> {
    fn into_slice(self, py: Python<'a>) -> &'a [T];
}

/// removes thin ReadOnlyCell wrapper
impl<'a, T> IntoSlice<'a, T> for PyBuffer<T>
    where T: Element
{
    fn into_slice(self, py: Python<'a>) -> &'a [T] {
        unsafe {
            let refcells = self.as_slice(py).unwrap_unchecked();  // assumes always recv contiguous buffers from py
            &*(refcells as *const [ReadOnlyCell<T>] as *const [T])
        }
    }
}



#[pymodule]
mod _core {
    use crate::IntoSlice;
    use crate::{ops, result::SimdResult};
    use pyo3::prelude::{pyfunction, Python};
    use pyo3::buffer::PyBuffer;

    #[pyfunction]
    fn add_f32(py: Python<'_>, a: PyBuffer<f32>, b: PyBuffer<f32>) -> SimdResult {
        ops::add_f32(a.into_slice(py), b.into_slice(py)).into()
    }

    #[pyfunction]
    fn mul_f32(py: Python<'_>, a: PyBuffer<f32>, b: PyBuffer<f32>) -> SimdResult {
        ops::mul_f32(a.into_slice(py), b.into_slice(py)).into()
    }

    #[pyfunction]
    fn neg_f32(py: Python<'_>, a: PyBuffer<f32>) -> SimdResult {
        ops::neg_f32(a.into_slice(py)).into()
    }

    #[pyfunction]
    fn sum_f32(py: Python<'_>, a: PyBuffer<f32>) -> f32 {
        ops::sum_f32(a.into_slice(py))
    }
}
