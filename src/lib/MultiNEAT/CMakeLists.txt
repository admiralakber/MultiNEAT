# EZMake MultiNEAT/src/lib/MultiNEAT/CMakeLists.txt
ez_unit_init(lib)

ez_this_unit_add_code(Random hh cc)
ez_this_unit_add_code(Traits hh cc)
ez_this_unit_add_code(Parameters hh cc)
ez_this_unit_add_code(Utils hh cc)
ez_this_unit_add_code(NeuralNetwork hh cc)
ez_this_unit_add_code(Species hh cc)
ez_this_unit_add_code(Innovation hh cc)
ez_this_unit_add_code(PhenotypeBehavior hh cc)
ez_this_unit_add_code(Population hh cc)
ez_this_unit_add_code(Genome hh cc)
ez_this_unit_add_code(Substrate hh cc)

ez_this_unit_add_header(Assert.hh)
ez_this_unit_add_header(Genes.hh)

ez_this_unit_build(SHARED)
ez_this_unit_link(PUBLIC ${Boost_LIBRARIES} pthread)

# Library unit testing
ez_this_unit_add_tests(test/Main.cc)
#ez_this_unit_link_tests(PUBLIC test_main)

ez_this_unit_install()
