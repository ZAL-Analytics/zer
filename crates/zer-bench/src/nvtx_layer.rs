/// `tracing` subscriber layer that maps span enter/exit to NVTX push/pop.
///
/// When compiled with `--features nvtx`, every `tracing` span that enters on a
/// thread pushes an NVTX range visible in Nsight Systems; dropping or exiting the
/// span pops it.  The layer is a zero-size unit type when the feature is absent.
///
/// # Thread safety
///
/// `nvtx::RangeGuard` is thread-local (not `Send`).  Guards are stored in a
/// thread-local `Vec` so they never cross thread boundaries.

#[cfg(feature = "nvtx")]
mod inner {
    use std::cell::RefCell;
    use tracing_subscriber::Layer;

    thread_local! {
        static GUARD_STACK: RefCell<Vec<zer_prof::NvtxGuard>> = RefCell::new(Vec::new());
    }

    pub struct NvtxLayer;

    impl<S> Layer<S> for NvtxLayer
    where
        S: tracing::Subscriber,
    {
        fn on_enter(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
            if let Some(span) = ctx.span(id) {
                let name = span.name().to_string();
                GUARD_STACK.with(|stack| {
                    stack.borrow_mut().push(zer_prof::NvtxGuard::new(&name));
                });
            }
        }

        fn on_exit(&self, _id: &tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
            GUARD_STACK.with(|stack| {
                stack.borrow_mut().pop();
            });
        }
    }
}

#[cfg(not(feature = "nvtx"))]
mod inner {
    #[allow(dead_code)]
    pub struct NvtxLayer;
}

#[allow(unused_imports)]
pub use inner::NvtxLayer;
