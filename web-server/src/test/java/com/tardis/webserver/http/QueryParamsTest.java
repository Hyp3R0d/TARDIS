package com.tardis.webserver.http;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class QueryParamsTest {

    @Test
    public void parsesSimplePairs() {
        QueryParams p = QueryParams.parse("msg=hello&count=3");
        assertEquals("hello", p.get("msg", ""));
        assertEquals(3, p.getInt("count", 0));
        assertTrue(p.has("msg"));
        assertTrue(p.has("count"));
    }

    @Test
    public void decodesUrlEncoding() {
        QueryParams p = QueryParams.parse("msg=%E4%BD%A0%E5%A5%BD");
        assertEquals("你好", p.get("msg", ""));
    }

    @Test
    public void handlesNullAndEmpty() {
        QueryParams p = QueryParams.parse(null);
        assertFalse(p.has("msg"));
        assertEquals("fallback", p.get("msg", "fallback"));
        assertEquals(7, p.getInt("msg", 7));
    }

    @Test
    public void handlesMalformedIntegers() {
        QueryParams p = QueryParams.parse("count=abc");
        assertEquals(9, p.getInt("count", 9));
    }
}
