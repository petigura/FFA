import pstats, cProfile
cProfile.runctx("import FFAtest", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
