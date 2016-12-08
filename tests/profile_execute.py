import cProfile


def call_function_to_profile():
    pass

pr = cProfile.Profile()
pr.enable()
call_function_to_profile()
pr.disable()
pr.dump_stats('profile_data.dump')
pr.print_stats(sort="calls")


# to view visualization of profiled data use command line tool "snakeviz filename"
