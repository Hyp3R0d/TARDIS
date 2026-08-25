package com.tardis.webserver.http;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

public final class QueryParams {

    private final Map<String, String> values;

    private QueryParams(Map<String, String> values) {
        this.values = values;
    }

    public static QueryParams parse(String rawQuery) {
        Map<String, String> map = new LinkedHashMap<>();
        if (rawQuery == null || rawQuery.isEmpty()) {
            return new QueryParams(map);
        }
        for (String pair : rawQuery.split("&")) {
            int eq = pair.indexOf('=');
            if (eq < 0) {
                map.put(decode(pair), "");
            } else {
                map.put(decode(pair.substring(0, eq)), decode(pair.substring(eq + 1)));
            }
        }
        return new QueryParams(map);
    }

    public String get(String name, String fallback) {
        return values.getOrDefault(name, fallback);
    }

    public int getInt(String name, int fallback) {
        try {
            return Integer.parseInt(values.getOrDefault(name, ""));
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    public boolean has(String name) {
        return values.containsKey(name);
    }

    public Map<String, String> asMap() {
        return Map.copyOf(values);
    }

    private static String decode(String s) {
        return URLDecoder.decode(s, StandardCharsets.UTF_8);
    }
}
