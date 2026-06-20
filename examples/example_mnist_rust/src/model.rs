#![allow(non_camel_case_types)]

use std::ffi::CString;
use std::ptr;

include!(concat!(env!("OUT_DIR"), "/nn.rs"));

pub struct Model {
    ptr: *mut NNModel,
    batch_size: usize,
}

impl Model {
    pub fn load(path: &str, batch_size: usize) -> Result<Self, i32> {
        let c_path = CString::new(path).unwrap();

        let mut model_ptr: *mut NNModel = ptr::null_mut();

        let result = unsafe { nnLoad(c_path.as_ptr(), &mut model_ptr, batch_size) };

        if result != 0 {
            return Err(result);
        }

        Ok(Model {
            ptr: model_ptr,
            batch_size,
        })
    }

    pub fn predict(&self, input: &[f32]) -> &[f32] {
        let mut out_size: usize = 0;
        let out_ptr = unsafe {
            nnPredict(
                self.ptr,
                input.as_ptr(),
                input.len() / self.batch_size,
                &mut out_size,
            )
        };

        unsafe { std::slice::from_raw_parts(out_ptr, out_size) }
    }

    pub fn predicted_classes(&self, pred: &[f32], num_classes: i32) -> Vec<Prediction> {
        let mut out = vec![Prediction { cls: 0, confidence: 0_f32 }; self.batch_size];
        unsafe {
            nnPredictedClasses(self.ptr, pred.as_ptr(), num_classes, out.as_mut_ptr());
        }

        out
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            nnFree(self.ptr);
        }
    }
}

pub struct NNMemoryGuard;

impl NNMemoryGuard {
    pub fn new() -> Self {
        unsafe {
            nnInit();
        }

        NNMemoryGuard
    }
}

impl Drop for NNMemoryGuard {
    fn drop(&mut self) {
        unsafe {
            nnCleanup();
        }
    }
}
