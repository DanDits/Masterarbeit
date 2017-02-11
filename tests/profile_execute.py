import cProfile
import importlib as im


def profile_function(function_to_profile, dump_file):
    pr = cProfile.Profile()
    pr.enable()
    print("Starting profiling...")
    function_to_profile()
    print("Stopping profiling...")
    pr.disable()
    pr.dump_stats(dump_file)
    pr.print_stats(sort="calls")


def profile_file(to_import_path, dump_file):
    pr = cProfile.Profile()
    pr.enable()
    print("Starting profiling...")
    im.import_module(to_import_path)
    print("Stopping profiling...")
    pr.disable()
    pr.dump_stats(dump_file)
    pr.print_stats(sort="calls")


if __name__ == "__main__":
    profile_file("demo.galerkin_show", "galerkin_slow")
# to view visualization of profiled data use command line tool "snakeviz filename"
