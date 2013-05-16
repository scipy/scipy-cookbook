`how to install
sunstudio <http://blogs.sun.com/migi/entry/opensolaris_howto_install_sun_studio>`__
and `build matplotlib on solaris
10 <http://blogs.sun.com/yongsun/entry/build_matplotlib_0_98_3>`__ might
give some hints.

JDH said:



.. code-block:: python

    > Hi Erik -- if you succeed, then we'll have convincing proof that
    > compiling mpl on solaris is easier than giving up the sauce.
    



Well, it has turned out to be easier than giving up the sauce (at least
for me), but only by a whisker. In the end, the fix is incredibly simple
(if you consider recompiling python and manually adjusting the
auto-produced pyconfig.h incredibly simple, anyway). After two solid
days of commenting this and that out, recompiling everything and its
mother 76 different ways from Sunday, poring over a legion of Solaris
sys includes, slaughtering a few spotlessly white lambs and one pure
black sheep, wrapping the bones and tendons and viscera in a double
layer of fat and burning the offering to Delphic Apollo, I found the
answer:

``1 download Python 2.4.2``

| ``2 after extracting it and running ./configure, edit the generated pyconfig.h as follows:``
| ``  ``
| ``    i) if _XOPEN_SOURCE is defined to be 600 (i.e., if the line "#define _XOPEN_SOURCE 600" appears in the file), redefine it to 500``

``    ii) if _XOPEN_SOURCE_EXTENDED is defined at all (i.e. if the line "#define _XOPEN_SOURCE_EXTENDED 1" appears in the file), comment out its definition``

``3 make && make install``

The problem was with Solaris's support for the X/Open standards. To make
a long story short, you can use Open Group Technical Standard, Issue 6
(XPG6/UNIX 03/SUSv3) (\_XOPEN\_SOURCE == 600) if and only if you are
using an ISO C99 compiler. If you use X/Open CAE Specification, Issue 5
(XPG5/UNIX 98/SUSv2) (\_XOPEN\_SOURCE == 500), you don't have to use an
ISO C99 compiler. For full details, see the Solaris header file
/usr/include/sys/feature\_tests.h.

This is why muhpubuh (AKA matplotlib---long story) compiles on Solaris
10 if you have the big bucks and can afford Sun's OpenStudio 10
compiler. gcc does not have full C99 support yet. In particular, its
lack of support for the wide character library makes the build go bust.
(See, e.g., http://gcc.gnu.org/c99status.html.)

More helpful links on the wchar problem with Python.h and Solaris :

``* ``\ ```http://lists.schmorp.de/pipermail/rxvt-unicode/2005q2/000092.html`` <http://lists.schmorp.de/pipermail/rxvt-unicode/2005q2/000092.html>`__

``* ``\ ```http://bugs.opensolaris.org/bugdatabase/view_bug.do?bug_id=6395191`` <http://bugs.opensolaris.org/bugdatabase/view_bug.do?bug_id=6395191>`__

``* ``\ ```http://mail.python.org/pipermail/patches/2005-June/017820.html`` <http://mail.python.org/pipermail/patches/2005-June/017820.html>`__

``* ``\ ```http://mail.python.org/pipermail/python-bugs-list/2005-November/030900.html`` <http://mail.python.org/pipermail/python-bugs-list/2005-November/030900.html>`__

CategoryCookbookMatplotlib

