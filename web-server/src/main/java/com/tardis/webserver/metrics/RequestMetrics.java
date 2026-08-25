package com.tardis.webserver.metrics;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public final class RequestMetrics {

    private final Instant startedAt = Instant.now();
    private final AtomicInteger activeRequests = new AtomicInteger();
    private final AtomicLong totalRequests = new AtomicLong();
    private final Map<String, AtomicInteger> perPath = new ConcurrentHashMap<>();

    public void requestStarted() {
        activeRequests.incrementAndGet();
        totalRequests.incrementAndGet();
    }

    public void requestFinished() {
        activeRequests.decrementAndGet();
    }

    public void countPath(String path) {
        perPath.computeIfAbsent(path, k -> new AtomicInteger()).incrementAndGet();
    }

    public long uptimeSeconds() {
        return Duration.between(startedAt, Instant.now()).getSeconds();
    }

    public int activeRequests() {
        return activeRequests.get();
    }

    public long totalRequests() {
        return totalRequests.get();
    }

    public Map<String, Integer> perPathCounts() {
        Map<String, Integer> snapshot = new ConcurrentHashMap<>();
        perPath.forEach((k, v) -> snapshot.put(k, v.get()));
        return snapshot;
    }
}
