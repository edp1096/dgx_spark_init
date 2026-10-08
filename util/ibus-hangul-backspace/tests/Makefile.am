# vim:set noet ts=4:
#
# ibus-hangul - The Hangul engine for IBus
#
# Copyright (c) 2019 Peng Wu <alexepico@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

TESTS = \
    hangul-backspace-tests \
    $(NULL)

CLEANFILES = $(test_metas)

EXTRA_DIST = $(test_metas_in) runtest

noinst_PROGRAMS = $(TESTS)

TESTS_ENVIRONMENT = \
    top_builddir=$(top_builddir) \
    top_srcdir=$(top_srcdir) \
    builddir=$(builddir) \
    srcdir=$(srcdir) \
    LD_LIBRARY_PATH="$(top_builddir)/src/.libs:$(top_builddir)/src" \
    DISABLE_GUI_TESTS="$(DISABLE_GUI_TESTS)" \
    $(NULL)

LOG_COMPILER = $(srcdir)/runtest
test_metas_in = meta.test.in

if ENABLE_INSTALLED_TESTS
test_execs = hangul-backspace-tests
test_metas = $(addsuffix .test, $(test_execs))
test_sources_DATA = $(test_metas)
test_sourcesdir = $(datadir)/installed-tests/hangul-backspace
test_execs_PROGRAMS = $(TESTS)
test_execsdir = $(libexecdir)/installed-tests/hangul-backspace
endif

hangul_backspace_tests_SOURCES = hangul-backspace-tests.c
hangul_backspace_tests_CFLAGS = \
        @IBUS_CFLAGS@ \
        @HANGUL_CFLAGS@ \
        @GTK_CFLAGS@ \
        -I$(top_srcdir)/src \
        -I$(top_builddir)/src \
        -DPKGDATADIR=\"$(pkgdatadir)\" \
        -DLOCALEDIR=\"$(localedir)\" \
        -DLIBEXECDIR=\"$(libexecdir)\" \
        -DIBUSHANGUL_DATADIR=\"$(datadir)/hangul-backspace\" \
        $(NULL)

hangul_backspace_tests_LDADD = \
        ../src/libinternal.a \
        @IBUS_LIBS@ \
        @HANGUL_LIBS@ \
        @GTK_LIBS@ \
        $(NULL)

$(test_metas): $(test_metas_in) $(test_programs)
	f=`echo $@ | sed -e 's/\.test//'`; \
	TEST_EXEC=$(test_execsdir)/$$f; \
	sed -e "s|@TEST_EXEC[@]|$$TEST_EXEC|g" $(test_metas_in) > $@.tmp; \
	mv $@.tmp $@; \
	$(NULL)
