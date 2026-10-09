# vim:set noet ts=4:
#
# ibus-hangul - The Hangul engine for IBus
#
# Copyright (c) 2007-2008 Huang Peng <shawn.p.huang@gmail.com>
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

AM_CFLAGS = \
	@IBUS_CFLAGS@ \
	@HANGUL_CFLAGS@ \
	-DPKGDATADIR=\"$(pkgdatadir)\" \
	$(NULL)
AM_LDFLAGS = \
	@IBUS_LIBS@ \
	@HANGUL_LIBS@ \
	$(NULL)

noinst_LIBRARIES = \
	libinternal.a \
	$(NULL)

libinternal_a_SOURCES = \
	engine.c \
	engine.h \
	ustring.c \
	ustring.h \
	i18n.h \
	$(NULL)

libinternal_a_CFLAGS = \
	@IBUS_CFLAGS@ \
	@HANGUL_CFLAGS@ \
	-DPKGDATADIR=\"$(pkgdatadir)\" \
	-DLOCALEDIR=\"$(localedir)\" \
	-DLIBEXECDIR=\"$(libexecdir)\" \
	-DIBUSHANGUL_DATADIR=\"$(datadir)/hangul-backspace\" \
	$(NULL)

check_PROGRAMS = \
	test-ustring \
	test-backspace \
	$(NULL)

TESTS = \
	$(check_PROGRAMS) \
	$(NULL)

AM_TESTS_ENVIRONMENT = \
	GSETTINGS_BACKEND=memory; export GSETTINGS_BACKEND; \
	GSETTINGS_SCHEMA_DIR="$(abs_builddir)/test-schemas"; export GSETTINGS_SCHEMA_DIR;

libexec_PROGRAMS = hangul-backspace

hangul_backspace_SOURCES = \
	main.c \
	$(NULL)

hangul_backspace_CFLAGS = \
	@IBUS_CFLAGS@ \
	@HANGUL_CFLAGS@ \
	-DPKGDATADIR=\"$(pkgdatadir)\" \
	-DLOCALEDIR=\"$(localedir)\" \
	-DLIBEXECDIR=\"$(libexecdir)\" \
	-DIBUSHANGUL_DATADIR=\"$(datadir)/hangul-backspace\" \
	$(NULL)

hangul_backspace_LDADD = \
	libinternal.a \
	@IBUS_LIBS@ \
	@HANGUL_LIBS@ \
	$(NULL)

component_DATA = \
	hangul-backspace.xml \
	$(NULL)
componentdir = @datadir@/ibus/component

EXTRA_DIST = \
	$(NULL)

CLEANFILES = \
	hangul-backspace.xml \
	test-schemas/gschemas.compiled \
	$(NULL)

hangul-backspace.xml: hangul-backspace.xml.in
	( \
		libexecdir=${libexecdir}; \
		pkgdatadir=${pkgdatadir}; \
		s=`cat $<`; \
		eval "echo \"$${s}\""; \
	) > $@

test: hangul-backspace
	$(builddir)/hangul-backspace

test_ustring_CFLAGS = $(IBUS_CFLAGS) $(HANGUL_CFLAGS)
test_ustring_LDADD = $(IBUS_LIBS)
test_ustring_SOURCES = test-ustring.c ustring.c ustring.h

test_backspace_CFLAGS = $(IBUS_CFLAGS) $(HANGUL_CFLAGS)
test_backspace_LDADD = libinternal.a $(IBUS_LIBS) $(HANGUL_LIBS) -ldl
test_backspace_SOURCES = test-backspace.c
test_backspace_DEPENDENCIES = libinternal.a test-schemas/gschemas.compiled

test-schemas/gschemas.compiled: $(top_srcdir)/data/org.freedesktop.ibus.engine.hangul-backspace.gschema.xml
	$(MKDIR_P) test-schemas
	glib-compile-schemas --strict --targetdir=test-schemas $(top_srcdir)/data

check-local:
		$(builddir)/test-ustring
