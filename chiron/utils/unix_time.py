
#Script from https://gist.github.com/turicas/5278558

from __future__ import absolute_import
from __future__ import print_function
from resource import getrusage as resource_usage, RUSAGE_SELF
from time import time as timestamp
from six.moves import range


def unix_time(function, args=tuple(), kwargs={}):
    '''Return `real`, `sys` and `user` elapsed time, like UNIX's command `time`
    You can calculate the amount of used CPU-time used by your
    function/callable by summing `user` and `sys`. `real` is just like the wall
    clock.
    Note that `sys` and `user`'s resolutions are limited by the resolution of
    the operating system's software clock (check `man 7 time` for more
    details).
    '''
    start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)
    function(*args, **kwargs)
    end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()

    return {'real': end_time - start_time,
            'sys': end_resources.ru_stime - start_resources.ru_stime,
            'user': end_resources.ru_utime - start_resources.ru_utime}


if __name__ == '__main__':
    def test(iterations):
        b = 1
        for i in range(iterations):
            b **= 2


    print("test")

    print((unix_time(test, (10,))))
    print((unix_time(test, (100,))))
    print((unix_time(test, (1000,))))
    print((unix_time(test, (10000,))))
    print((unix_time(test, (100000,))))
    print((unix_time(test, (1000000,))))
