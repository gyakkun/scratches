import java.util.concurrent.locks.AbstractQueuedSynchronizer;
import java.util.concurrent.locks.ReentrantLock;

class Scratch {
    public static void main(String[] args) {
        // 多线程测试
        Thread t1 = new Thread(new ReentrantLockTryLockTest(1));
        Thread t2 = new Thread(new ReentrantLockTryLockTest(2));
        t1.start();
        t2.start();
    }

}

class ReentrantLockTryLockTest implements Runnable {
    public static ReentrantLock r1 = new ReentrantLock();
    public static ReentrantLock r2 = new ReentrantLock();

    int lock;

    public ReentrantLockTryLockTest(int lock) {
        this.lock = lock;
    }


    @Override
    public void run() {
        if (lock == 1) {
            while (true) {
                if (r1.tryLock()) {
                    try {
                        try {
                            Thread.sleep(500);
                        } catch (InterruptedException ie) {
                            ;
                        }
                        if (r2.tryLock()) {
                            // 加锁后的try catch写法
                            try {
                                System.out.println(Thread.currentThread().getId() + ": done.");
                                return;
                            } finally {
                                r2.unlock();

                            }
                        }
                    } finally {
                        r1.unlock();
                    }
                }
            }
        } else {
            while (true) {
                if (r2.tryLock()) {
                    try {
                        try {
                            Thread.sleep(500);
                        } catch (InterruptedException ie) {
                            ;
                        }
                        if (r1.tryLock()) {
                            // 加锁后的try catch写法
                            try {
                                System.out.println(Thread.currentThread().getId() + ": done.");
                                return;
                            } finally {
                                r1.unlock();

                            }
                        }
                    } finally {
                        r2.unlock();
                    }
                }
            }
        }
    }
}