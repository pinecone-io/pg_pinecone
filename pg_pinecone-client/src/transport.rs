//! A read that a signal interrupts resumes instead of failing. Linux does not restart a blocking
//! read on a socket with a timeout, even under `SA_RESTART`, and a failed request is sent again.

use std::{io, time::Instant};

use ureq::{
    Error,
    unversioned::transport::{
        Buffers, ConnectionDetails, Connector, NextTimeout, Transport, time::Duration,
    },
};

/// Wraps each transport the default connector makes.
#[derive(Debug)]
pub(crate) struct ResumeReads;

impl Connector<Box<dyn Transport>> for ResumeReads {
    type Out = Resuming;

    fn connect(
        &self,
        _: &ConnectionDetails,
        chained: Option<Box<dyn Transport>>,
    ) -> Result<Option<Resuming>, Error> {
        Ok(chained.map(Resuming))
    }
}

#[derive(Debug)]
pub(crate) struct Resuming(Box<dyn Transport>);

impl Transport for Resuming {
    fn buffers(&mut self) -> &mut dyn Buffers {
        self.0.buffers()
    }

    fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> Result<(), Error> {
        self.0.transmit_output(amount, timeout)
    }

    fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, Error> {
        let started = Instant::now();
        let mut next = timeout;
        loop {
            match self.0.await_input(next) {
                Err(Error::Io(e)) if e.kind() == io::ErrorKind::Interrupted => {}
                done => return done,
            }
            if let Duration::Exact(total) = timeout.after {
                match total.checked_sub(started.elapsed()) {
                    Some(left) if !left.is_zero() => next.after = Duration::Exact(left),
                    _ => return Err(Error::Timeout(timeout.reason)),
                }
            }
        }
    }

    fn is_open(&mut self) -> bool {
        self.0.is_open()
    }

    fn is_tls(&self) -> bool {
        self.0.is_tls()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
        thread,
        time::Duration as Wait,
    };

    use ureq::{Timeout, unversioned::transport::LazyBuffers};

    use super::*;

    /// Answers each wait with the next scripted error, then with progress.
    #[derive(Debug)]
    struct Script {
        buffers: LazyBuffers,
        errors: VecDeque<io::ErrorKind>,
        pause: Wait,
        waits: Arc<Mutex<Vec<NextTimeout>>>,
    }

    impl Transport for Script {
        fn buffers(&mut self) -> &mut dyn Buffers {
            &mut self.buffers
        }

        fn transmit_output(&mut self, _: usize, _: NextTimeout) -> Result<(), Error> {
            Ok(())
        }

        fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, Error> {
            self.waits.lock().expect("waits").push(timeout);
            thread::sleep(self.pause);
            match self.errors.pop_front() {
                Some(kind) => Err(Error::Io(kind.into())),
                None => Ok(true),
            }
        }

        fn is_open(&mut self) -> bool {
            true
        }
    }

    fn wrapped(errors: &[io::ErrorKind], pause: Wait) -> (Resuming, Arc<Mutex<Vec<NextTimeout>>>) {
        let waits = Arc::new(Mutex::new(Vec::new()));
        let script = Script {
            buffers: LazyBuffers::new(64, 64),
            errors: errors.iter().copied().collect(),
            pause,
            waits: waits.clone(),
        };
        (Resuming(Box::new(script)), waits)
    }

    fn within(secs: u64) -> NextTimeout {
        NextTimeout {
            after: Duration::from_secs(secs),
            reason: Timeout::Global,
            per_read: None,
            per_write: None,
        }
    }

    #[test]
    fn an_interrupted_read_is_read_again_within_the_same_deadline() {
        let interrupted = [io::ErrorKind::Interrupted; 2];
        let (mut transport, waits) = wrapped(&interrupted, Wait::from_millis(5));

        assert!(transport.await_input(within(5)).expect("progress"));

        let waits = waits.lock().expect("waits");
        assert_eq!(waits.len(), 3, "two interruptions, then the read");
        assert!(waits.windows(2).all(|w| w[1].after < w[0].after));
        assert!(waits.iter().all(|w| w.after <= Duration::from_secs(5)));
    }

    #[test]
    fn any_other_error_is_handed_back_at_once() {
        let (mut transport, waits) = wrapped(&[io::ErrorKind::ConnectionReset], Wait::ZERO);

        let err = transport.await_input(within(5)).expect_err("reset");

        assert!(matches!(err, Error::Io(e) if e.kind() == io::ErrorKind::ConnectionReset));
        assert_eq!(waits.lock().expect("waits").len(), 1);
    }

    #[test]
    fn interruptions_do_not_outlast_the_deadline() {
        let interrupted = [io::ErrorKind::Interrupted; 100];
        let (mut transport, waits) = wrapped(&interrupted, Wait::from_millis(20));
        let timeout = NextTimeout {
            after: Duration::from_millis(50),
            reason: Timeout::Global,
            per_read: None,
            per_write: None,
        };

        let err = transport.await_input(timeout).expect_err("timeout");

        assert!(matches!(err, Error::Timeout(Timeout::Global)));
        assert!(waits.lock().expect("waits").len() < 100);
    }
}
