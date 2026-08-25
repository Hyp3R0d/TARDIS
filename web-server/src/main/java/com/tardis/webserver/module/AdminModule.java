package com.tardis.webserver.module;

import com.tardis.webserver.core.RouteModule;
import com.tardis.webserver.core.Router;
import com.tardis.webserver.http.Json;
import com.tardis.webserver.metrics.RequestMetrics;
import com.tardis.webserver.util.Bytes;

public final class AdminModule implements RouteModule {

    private final RequestMetrics metrics;

    public AdminModule(RequestMetrics metrics) {
        this.metrics = metrics;
    }

    @Override
    public void register(Router router) {
        router.register("GET", "/admin/info", ctx -> ctx.json(200, Json.of(
                "role", "admin",
                "message", "protected area")));

        router.register("GET", "/admin/metrics", ctx -> ctx.json(200, Json.of(
                "totalRequests", metrics.totalRequests(),
                "activeRequests", metrics.activeRequests(),
                "uptimeSeconds", metrics.uptimeSeconds(),
                "memory", Bytes.humanReadable(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()),
                "paths", metrics.perPathCounts())));
    }
}
