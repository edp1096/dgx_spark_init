/* Regression tests; no keys or settings reach the user's desktop. */
#include <ibus.h>
#include "engine.h"

static GDBusConnection *connection;
static GArray *forwarded;
static GString *committed;

typedef struct {
    guint keyval;
    guint keycode;
    guint modifiers;
} ForwardedKey;

typedef struct {
    GSettings *settings;
    IBusEngine *engine;
} Fixture;

/* Interpose outgoing calls while exercising the real engine implementation. */
void
ibus_engine_forward_key_event (IBusEngine *engine, guint keyval,
                               guint keycode, guint modifiers)
{
    ForwardedKey event = { keyval, keycode, modifiers };
    g_array_append_val (forwarded, event);
}

void
ibus_engine_commit_text (IBusEngine *engine, IBusText *text)
{
    g_string_append (committed, ibus_text_get_text (text));
    g_object_ref_sink (text);
    g_object_unref (text);
}

static gboolean
key (Fixture *fixture, guint keyval, guint modifiers)
{
    return IBUS_ENGINE_GET_CLASS (fixture->engine)->process_key_event (
        fixture->engine, keyval, keyval == IBUS_BackSpace ? 14 : 0, modifiers);
}

static void
setup (Fixture *fixture, gconstpointer data)
{
    fixture->settings = g_settings_new ("org.freedesktop.ibus.engine.hangul-backspace");
    g_settings_reset (fixture->settings, "auto-reorder");
    g_settings_set_string (fixture->settings, "initial-input-mode", "hangul");
    g_settings_set_string (fixture->settings, "preedit-mode", "syllable");
    g_settings_set_boolean (fixture->settings, "use-event-forwarding",
                            GPOINTER_TO_INT (data));
    ibus_hangul_init (NULL);
    fixture->engine = ibus_engine_new_with_type (
        IBUS_TYPE_HANGUL_ENGINE, "backspace-test", "/org/test/Engine", connection);
    g_object_ref_sink (fixture->engine);
    g_array_set_size (forwarded, 0);
    g_string_truncate (committed, 0);
#if IBUS_CHECK_VERSION(1, 5, 28)
    IBUS_ENGINE_GET_CLASS (fixture->engine)->set_content_type (
        fixture->engine, IBUS_INPUT_PURPOSE_TERMINAL, 0);
#endif
    IBUS_ENGINE_GET_CLASS (fixture->engine)->set_capabilities (
        fixture->engine, IBUS_CAP_PREEDIT_TEXT | IBUS_CAP_FOCUS);
}

static void
teardown (Fixture *fixture, gconstpointer data)
{
    ibus_object_destroy (IBUS_OBJECT (fixture->engine));
    g_object_unref (fixture->engine);
    ibus_hangul_exit ();
    g_object_unref (fixture->settings);
}

static void
empty_composition (Fixture *fixture)
{
    g_assert_true (key (fixture, IBUS_r, 0));
    g_assert_true (key (fixture, IBUS_k, 0));
    g_assert_true (key (fixture, IBUS_BackSpace, 0));
    g_assert_true (key (fixture, IBUS_BackSpace, 0));
    g_assert_cmpuint (forwarded->len, ==, 0);
    g_assert_cmpuint (committed->len, ==, 0);
}

#if IBUS_CHECK_VERSION(1, 5, 28)
static void
test_terminal_repeats (Fixture *fixture, gconstpointer data)
{
    empty_composition (fixture);
    for (guint i = 0; i < 5; i++)
        g_assert_true (key (fixture, IBUS_BackSpace, 0));
    g_assert_cmpstr (committed->str, ==, "\177\177\177\177\177");
    g_assert_cmpuint (forwarded->len, ==, 0);
    g_assert_false (key (fixture, IBUS_BackSpace, IBUS_RELEASE_MASK));
    g_assert_cmpuint (committed->len, ==, 5);
    /* A fresh press goes through the terminal's own Backspace mapping. */
    g_assert_false (key (fixture, IBUS_BackSpace, 0));
    g_assert_cmpuint (committed->len, ==, 5);
}

static void
test_reset (Fixture *fixture, gconstpointer data)
{
    empty_composition (fixture);
    IBUS_ENGINE_GET_CLASS (fixture->engine)->reset (fixture->engine);
    g_assert_false (key (fixture, IBUS_BackSpace, 0));
    g_assert_cmpuint (committed->len, ==, 0);
}
#endif

static void
assert_forwarded (guint count)
{
    g_assert_cmpuint (forwarded->len, ==, count);
    g_assert_cmpuint (committed->len, ==, 0);
    for (guint i = 0; i < count; i++) {
        ForwardedKey event = g_array_index (forwarded, ForwardedKey, i);
        g_assert_cmpuint (event.keyval, ==, IBUS_BackSpace);
        g_assert_cmpuint (event.keycode, ==, 14);
        g_assert_cmpuint (event.modifiers, ==, 0);
    }
}

static void
test_nonterminal (Fixture *fixture, gconstpointer data)
{
    IBUS_ENGINE_GET_CLASS (fixture->engine)->set_content_type (
        fixture->engine, IBUS_INPUT_PURPOSE_FREE_FORM, 0);
    empty_composition (fixture);
    for (guint i = 0; i < 5; i++)
        g_assert_true (key (fixture, IBUS_BackSpace, 0));
    assert_forwarded (5);
}

#if IBUS_CHECK_VERSION(1, 5, 28)
static void
test_sync_client (Fixture *fixture, gconstpointer data)
{
    IBUS_ENGINE_GET_CLASS (fixture->engine)->set_capabilities (
        fixture->engine, IBUS_CAP_SYNC_PROCESS_KEY_V2);
    empty_composition (fixture);
    for (guint i = 0; i < 5; i++)
        g_assert_true (key (fixture, IBUS_BackSpace, 0));
    assert_forwarded (5);
}
#endif

static void
test_empty_context (Fixture *fixture, gconstpointer data)
{
    for (guint i = 0; i < 5; i++)
        g_assert_false (key (fixture, IBUS_BackSpace, 0));
    assert_forwarded (0);
}

static void
test_latin_mode (Fixture *fixture, gconstpointer data)
{
    g_assert_true (key (fixture, IBUS_Hangul, 0));
    test_empty_context (fixture, data);
}

static void
test_control_backspace (Fixture *fixture, gconstpointer data)
{
    empty_composition (fixture);
    g_assert_false (key (fixture, IBUS_BackSpace, IBUS_CONTROL_MASK));
    assert_forwarded (0);
    g_assert_true (key (fixture, IBUS_Hangul, 0));
    g_assert_false (key (fixture, IBUS_BackSpace, IBUS_CONTROL_MASK));
    assert_forwarded (0);
}

static void
test_jamo_order (Fixture *fixture, gconstpointer data)
{
    g_assert_true (key (fixture, IBUS_l, 0));
    g_assert_true (key (fixture, IBUS_s, 0));
    key (fixture, IBUS_space, 0);
    g_assert_cmpstr (committed->str, ==, "ㅣㄴ");
}

static void
test_normal_composition (Fixture *fixture, gconstpointer data)
{
    g_assert_true (key (fixture, IBUS_s, 0));
    g_assert_true (key (fixture, IBUS_l, 0));
    key (fixture, IBUS_space, 0);
    g_assert_cmpstr (committed->str, ==, "니");
}

static void
test_reorder_opt_in (Fixture *fixture, gconstpointer data)
{
    g_settings_set_boolean (fixture->settings, "auto-reorder", TRUE);
    g_assert_true (key (fixture, IBUS_l, 0));
    g_assert_true (key (fixture, IBUS_s, 0));
    key (fixture, IBUS_space, 0);
    g_assert_cmpstr (committed->str, ==, "니");
}

int
main (int argc, char **argv)
{
    GTestDBus *bus;
    GError *error = NULL;
    int result;

    g_test_init (&argc, &argv, NULL);
    g_setenv ("GSETTINGS_BACKEND", "memory", TRUE);
    ibus_init ();
    bus = g_test_dbus_new (G_TEST_DBUS_NONE);
    g_test_dbus_up (bus);
    connection = g_dbus_connection_new_for_address_sync (
        g_test_dbus_get_bus_address (bus),
        G_DBUS_CONNECTION_FLAGS_AUTHENTICATION_CLIENT |
        G_DBUS_CONNECTION_FLAGS_MESSAGE_BUS_CONNECTION, NULL, NULL, &error);
    g_assert_no_error (error);
    forwarded = g_array_new (FALSE, FALSE, sizeof (ForwardedKey));
    committed = g_string_new (NULL);

#define ADD_TEST(name, func, forwarding) \
    g_test_add ("/backspace/" name, Fixture, GINT_TO_POINTER (forwarding), \
                setup, func, teardown)
#if IBUS_CHECK_VERSION(1, 5, 28)
    ADD_TEST ("terminal-repeats", test_terminal_repeats, TRUE);
    ADD_TEST ("terminal-without-forwarding", test_terminal_repeats, FALSE);
    ADD_TEST ("reset", test_reset, TRUE);
    ADD_TEST ("sync-client", test_sync_client, TRUE);
#endif
    ADD_TEST ("nonterminal", test_nonterminal, TRUE);
    ADD_TEST ("empty-context", test_empty_context, TRUE);
    ADD_TEST ("latin-mode", test_latin_mode, TRUE);
    ADD_TEST ("control-shortcut", test_control_backspace, TRUE);
    ADD_TEST ("jamo-order", test_jamo_order, TRUE);
    ADD_TEST ("normal-composition", test_normal_composition, TRUE);
    ADD_TEST ("reorder-opt-in", test_reorder_opt_in, TRUE);
    result = g_test_run ();

    g_string_free (committed, TRUE);
    g_array_unref (forwarded);
    g_dbus_connection_close_sync (connection, NULL, NULL);
    g_object_unref (connection);
    g_test_dbus_down (bus);
    g_object_unref (bus);
    return result;
}
