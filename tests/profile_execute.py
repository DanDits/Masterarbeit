import cProfile


def profile_function(function_to_profile, dump_file):
    pr = cProfile.Profile()
    pr.enable()
    function_to_profile()
    pr.disable()
    pr.dump_stats(dump_file)
    pr.print_stats(sort="calls")


# to view visualization of profiled data use command line tool "snakeviz filename"
