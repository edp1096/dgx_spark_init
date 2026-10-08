/* vim: set et sts=4: */
/* ibus-hangul - The Hangul Engine For IBus
 * Copyright (C) 2018 Choe Hwanjin <choe.hwanjin@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "ustring.h"

#include <glib.h>


static void
test_ustring_compare(void)
{
    UString* s1 = ustring_new();
    UString* s2 = ustring_new();

    ustring_append_utf8(s1, "abc");
    ustring_append_utf8(s2, "abd");
    g_assert_cmpint(ustring_compare(s1, s2), <, 0);

    ustring_clear(s1);
    ustring_clear(s2);

    ustring_append_utf8(s1, "abc");
    ustring_append_utf8(s2, "abb");
    g_assert_cmpint(ustring_compare(s1, s2), >, 0);

    ustring_clear(s1);
    ustring_clear(s2);

    ustring_append_utf8(s1, "abc");
    ustring_append_utf8(s2, "abc");
    g_assert_cmpint(ustring_compare(s1, s2), ==, 0);

    ustring_delete(s1);
    ustring_delete(s2);
}

int
main(int argc, char* argv[])
{
    g_test_init(&argc, &argv, NULL);

    g_test_add_func("/hangul-backspace/ustring/compare", test_ustring_compare);

    int result = g_test_run();
    return result;
}
