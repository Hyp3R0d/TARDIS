package com.tardis.webserver.util;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class BytesTest {

    @Test
    public void formatsUnits() {
        assertEquals("512 B", Bytes.humanReadable(512));
        assertEquals("1.5 KB", Bytes.humanReadable(1536));
        assertEquals("2.0 MB", Bytes.humanReadable(2 * 1024 * 1024));
        assertEquals("1.0 GB", Bytes.humanReadable(1024L * 1024 * 1024));
    }

    @Test
    public void handlesZero() {
        assertEquals("0 B", Bytes.humanReadable(0));
    }
}
