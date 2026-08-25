package com.tardis.webserver.http;

import org.junit.Test;

import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;

public class MimeTypesTest {

    @Test
    public void resolvesKnownExtensions() {
        assertEquals("text/html; charset=utf-8", MimeTypes.forFile(Paths.get("index.html")));
        assertEquals("text/css; charset=utf-8", MimeTypes.forFile(Paths.get("style.css")));
        assertEquals("application/javascript; charset=utf-8", MimeTypes.forFile(Paths.get("app.js")));
        assertEquals("image/png", MimeTypes.forFile(Paths.get("logo.png")));
        assertEquals("application/json; charset=utf-8", MimeTypes.forFile(Paths.get("data.json")));
    }

    @Test
    public void isCaseInsensitive() {
        assertEquals("image/jpeg", MimeTypes.forFile(Paths.get("photo.JPG")));
    }

    @Test
    public void fallsBackToOctetStream() {
        assertEquals(MimeTypes.DEFAULT, MimeTypes.forFile(Paths.get("unknown.xyz")));
        assertEquals(MimeTypes.DEFAULT, MimeTypes.forFile(Paths.get("no-extension")));
    }
}
