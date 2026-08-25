package com.tardis.webserver.http;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class JsonTest {

    @Test
    public void encodesStringsAndNumbers() {
        assertEquals("{\"status\":\"UP\",\"uptimeSeconds\":42}", Json.of("status", "UP", "uptimeSeconds", 42L));
    }

    @Test
    public void escapesQuotesAndBackslashes() {
        assertEquals("{\"echo\":\"a\\\"b\\\\c\"}", Json.of("echo", "a\"b\\c"));
    }

    @Test
    public void handlesBooleans() {
        assertEquals("{\"ok\":true,\"fail\":false}", Json.of("ok", true, "fail", false));
    }
}
