package com.tardis.webserver.util;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class TokenBucketTest {

    @Test
    public void acquiresUpToCapacityThenRejects() {
        TokenBucket bucket = new TokenBucket(3.0, 1.0);
        assertTrue(bucket.tryAcquire());
        assertTrue(bucket.tryAcquire());
        assertTrue(bucket.tryAcquire());
        assertFalse(bucket.tryAcquire());
    }

    @Test
    public void refillsOverTime() throws InterruptedException {
        TokenBucket bucket = new TokenBucket(1.0, 1000.0);
        assertTrue(bucket.tryAcquire());
        assertFalse(bucket.tryAcquire());
        Thread.sleep(20);
        assertTrue(bucket.tryAcquire());
    }
}
