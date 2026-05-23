Reporting Bugs and Feature Requests
=====================================

All issues are tracked on the GitHub repository:

`github.com/ZAL-Analytics/zer <https://github.com/ZAL-Analytics/zer>`_

Before opening an issue
------------------------

Search the existing issues first,the problem may already have a thread
with a workaround or a fix on the way. The issue tracker is the right
place for bugs, feature requests, and documentation gaps. For questions
about using zer in your own project see :doc:`contact`.

Bug reports
------------

A useful bug report answers three questions: what did you expect to happen,
what actually happened, and how can someone reproduce it?

Please include:

* The zer version (``zer = "x.y.z"`` from your ``Cargo.toml``).
* The Rust toolchain version (``rustc --version``).
* The operating system and, if relevant, GPU model and driver version.
* The feature flags enabled (``pipeline``, ``cuda``, ``judge_cpu``, etc.).
* A minimal ``Cargo.toml`` and code snippet that reproduces the problem.
* The full error message or unexpected output, including any panic backtraces.

For panics, run with ``RUST_BACKTRACE=1`` to capture the full trace:

.. code-block:: bash

   RUST_BACKTRACE=1 cargo run --example your_example

If the problem involves incorrect entity resolution results (wrong matches,
missed matches), attach a small anonymised excerpt of the input data and the
expected vs. actual ``ClusterView`` output. Even 20–50 records that reproduce
the error are enough.

Feature requests
-----------------

Open an issue with the ``enhancement`` label and describe:

* The use case,what are you trying to do that zer does not currently support?
* Why existing features do not cover it (e.g. the built-in blocking keys do
  not handle your domain, or the ``EntityStore`` trait is missing a method you
  need).
* An example API sketch if you have one in mind.

Documentation issues
---------------------

If something in these docs is wrong, unclear, or missing, open an issue
with the ``documentation`` label. A short quote of the confusing text and
a sentence on what you expected it to say is sufficient.

Pull requests
--------------

Small, focused pull requests are easiest to review. For any non-trivial
change, open an issue first to discuss the approach before writing code.
The repository does not yet have a formal ``CONTRIBUTING.md``,if you are
unsure whether a change is in scope, ask in the issue before starting work.
