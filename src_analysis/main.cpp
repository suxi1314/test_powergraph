/*
 * Copyright (c) 2009 Carnegie Mellon University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 *
 */
/**
 * \file main.cpp
 * This is a copy of the main function of cgs_lda.cpp for analysis the code.
 * cgs_lda.cpp的主要思想如下：
 * graph
 * 把输入的doc_word_count文件中的矩阵存为二部图
 * 左侧的节点是矩阵第一列doc，右侧的节点是矩阵第二列word
 * doc_word_count中的每一行代表某个word在某个doc中出现了times次
 * 在图中表示为左侧的某个doc指向右侧的某个word的一条边edge
 * edge上存储的是一个长度为times的vector(occur_time, topic)
 *  word节点和doc节点上存储的数据是topic被分配到word或doc的次数count的vector(topic, count)
 * 由图可知，只有入边的是word节点，只有出边的是doc节点

doc_vertex                                     edge                             word_vertex


                                                           .  .  .  .  .> word1------vector(topic, count)
                                                       .
                                                   .
                                               .
                                           . vector(occur_time, topic)
                                       .
                                   .
vector(topic, count)------doc1 .
                                   .
                                       .
                                           . vector(occur_time, topic)
                                               .
                                                   .
                                                       .
                                                           .  .  .  .  .> word2------vector(topic, count)
                                                       .
                                                   .
                                               .
                                           . vector(occur_time, topic)
                                       .
                                   .
vector(topic, count)------doc2 .
                                   .
                                       .
                                           . vector(occur_time, topic)
                                               .
                                                   .
                                                       .
                                                           .  .  .  .  .> word3------vector(topic, count)



 * gibbs采样的迭代过程：
 * 论文Finding Scientific Topics所述，需要通过Gibbs采样计算出p(w|z), p(z), p(w,z)的likelihood
 *
 * init
 * Gibbs采样的初始化，需要向每个doc中的每个word的每次出现分配一个随机的topic
 * 也就是向每个edge上的vector的每个元素分配新的topic，分配完成后，
 *
 * gather
 * word节点统计到该次迭代为止，该word被分配到每个topic的次数，存在word节点的vector(topic,count)中
 * doc节点统计到该次迭代为止，该doc中的所有word被分配到每个topic的次数，存在doc节点的vector(topic,count)中
 *
 * aggragator(map, finalize )
 * 得到本次迭代的word-topic矩阵和doc-topic矩阵
 * 计算本次迭代的p(w|z), p(z), p(w,z)的likelihood（注意gamma函数取了对数，原来公式中的乘除变为加减）
 *
 * scatter
 * 预测并向每个edge上的vector的每个元素分配新的topic
 *
 * 迭代...
 *
 * \author Shanshan Wang
 */

int main(int argc, char** argv) {
  global_logger().set_log_level(LOG_INFO);
  global_logger().set_log_to_console(true);
  ///! Initialize control plain using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;
  //  INITIALIZE_EVENT_LOG(dc);
  ADD_CUMULATIVE_EVENT(TOKEN_CHANGES, "Token Changes", "Changes");

  // Parse command line options -----------------------------------------------
  const std::string description =
    "\n=========================================================================\n"
    "The Collapsed Gibbs Sampler for the LDA model implements\n"
    "a highly asynchronous version of parallel LDA in which document\n"
    "and word counts are maintained in an eventually consistent\n"
    "manner.\n"
    "\n"
    "The standard usage is: \n"
    "\t./cgs_lda --dictionary dictionary.txt --corpus doc_word_count.tsv\n"
    "where dictionary.txt contains: \n"
    "\taaa \n\taaai \n\tabalone \n\t   ... \n"
    "each line number corresponds to wordid (i.e aaa has wordid=0)\n\n"
    "and doc_word_count.tsv is formatted <docid> <wordid> <count>:\n"
    "(where wordid is indexed starting from zero and docid are positive integers)\n"
    "\t0\t0\t3\n"
    "\t0\t5\t1\n"
    "\t ...\n\n"
    "For JSON format, make sure docid are negative integers index starting from -2 \n\n"
    "To learn more about the NLP package and its applications visit\n\n"
    "\t\t http://graphlab.org \n\n"
    "Additional Options";
  graphlab::command_line_options clopts(description);
  std::string corpus_dir;
  std::string dictionary_fname;
  std::string doc_dir;
  std::string word_dir;
  std::string exec_type = "asynchronous";
  std::string format = "matrix";

  clopts.attach_option("dictionary", dictionary_fname,
                       "The file containing the list of unique words");
  clopts.attach_option("engine", exec_type,
                       "The engine type synchronous or asynchronous");
  clopts.attach_option("corpus", corpus_dir,
                       "The directory or file containing the corpus data.");
  clopts.add_positional("corpus");
  clopts.attach_option("ntopics", NTOPICS,
                       "Number of topics to use.");
  clopts.attach_option("alpha", ALPHA,
                       "The document hyper-prior");
  clopts.attach_option("beta", BETA,
                       "The word hyper-prior");
  clopts.attach_option("topk", TOPK,
                       "The number of words to report");
  clopts.attach_option("interval", INTERVAL,
                       "statistics reporting interval (in seconds)");
  clopts.attach_option("lik_interval", LIK_INTERVAL,
                       "likelihood reporting interval (in seconds)");
  clopts.attach_option("max_count", MAX_COUNT,
                       "The maximum number of occurences of a word in a document.");
  clopts.attach_option("format", format,
                       "Formats: matrix,json,json-gzip");
  clopts.attach_option("burnin", BURNIN,
                       "The time in second to run until a sample is collected. "
                       "If less than zero the sampler runs indefinitely.");
  clopts.attach_option("doc_dir", doc_dir,
                       "The output directory to save the final document counts.");
  clopts.attach_option("word_dir", word_dir,
                       "The output directory to save the final words counts.");


  if(!clopts.parse(argc, argv)) {
    graphlab::mpi_tools::finalize();
    return clopts.is_set("help")? EXIT_SUCCESS : EXIT_FAILURE;
  }

  if(dictionary_fname.empty()) {
    logstream(LOG_WARNING) << "No dictionary file was provided." << std::endl
                           << "Top k words will not be estimated." << std::endl;
  }

  if(corpus_dir.empty()) {
    logstream(LOG_ERROR) << "No corpus file was provided." << std::endl;
    return EXIT_FAILURE;
  }

  // Start the webserver
  graphlab::launch_metric_server();
  graphlab::add_metric_server_callback("wordclouds", word_cloud_callback);


  ///! Initialize global variables
  GLOBAL_TOPIC_COUNT.resize(NTOPICS);
  if(!dictionary_fname.empty()) {
    const bool success = load_dictionary(dictionary_fname);
    if(!success) {
      logstream(LOG_ERROR) << "Error loading dictionary." << std::endl;
      return EXIT_FAILURE;
    }
  }

  if(ALPHA <= 0) {
    logstream(LOG_ERROR)
      << "Alpha must be positive (alpha=" << ALPHA << ")!"  << std::endl;
    return EXIT_FAILURE;
  }

  if(BETA <= 0) {
    logstream(LOG_ERROR)
      << "Beta must be positive (beta=" << BETA << ")!"  << std::endl;
    return EXIT_FAILURE;
  }

  /// Initialize the log_gamma precached calculations.
  ALPHA_LGAMMA.init(ALPHA, 100000);
  BETA_LGAMMA.init(BETA, 1000000);


  ///! load the graph
  graph_type graph(dc, clopts);
  {
    const bool success =
      load_and_initialize_graph(dc, graph, corpus_dir, format);
    if(!success) {
      logstream(LOG_ERROR) << "Error loading graph." << std::endl;
      return EXIT_FAILURE;
    }
  }


  const size_t ntokens = graph.map_reduce_edges<size_t>(count_tokens);
  dc.cout() << "Total tokens: " << ntokens << std::endl;


  /**
  * You can check the function add_vertex_aggregator in the doc:
  * test_powergraph-master/doc/classgraphlab_1_1iengine.html#a39c802e7271358becf2cf2b2418b943a
  * This is a simple description of the function:
  * bool graphlab::iengine< VertexProgram >::add_vertex_aggregator 	( 	const std::string &  	key,
	*                                                                     VertexMapType  	map_function,
  *                                                                   	FinalizerType  	finalize_function)
  * Creates a vertex aggregator associated to a particular key.
  * The map_function is called over every vertex in the graph, and the return value of the map is summed.
  * The finalize_function is then called on the result of the reduction. The finalize_function is called on all machines.
  * The map_function should only read the graph data, and should not make any modifications.
  */
  engine_type engine(dc, graph, exec_type, clopts);
  ///! Add an aggregator
  if(!DICTIONARY.empty()) {
    const bool success =
      engine.add_vertex_aggregator<topk_aggregator>
      ("topk", topk_aggregator::map, topk_aggregator::finalize) &&
      engine.aggregate_periodic("topk", INTERVAL);
    ASSERT_TRUE(success);
  }

  { // Add the Global counts aggregator
    const bool success =
      engine.add_vertex_aggregator<factor_type>
      ("global_counts",
       global_counts_aggregator::map,
       global_counts_aggregator::finalize) &&
      engine.aggregate_periodic("global_counts", 5);
    ASSERT_TRUE(success);
  }

  { // Add the likelihood aggregator
    const bool success =
      engine.add_vertex_aggregator<likelihood_aggregator>
      ("likelihood",
       likelihood_aggregator::map,
       likelihood_aggregator::finalize) &&
      engine.aggregate_periodic("likelihood", LIK_INTERVAL);
    ASSERT_TRUE(success);
  }

  ///! schedule only documents
  dc.cout() << "Running The Collapsed Gibbs Sampler" << std::endl;
  engine.map_reduce_vertices<graphlab::empty>(signal_only::docs);
  graphlab::timer timer;
  // Enable sampling
  cgs_lda_vertex_program::DISABLE_SAMPLING = false;
  // Run the engine
  engine.start();
  // Finalize the counts
  cgs_lda_vertex_program::DISABLE_SAMPLING = true;
  engine.signal_all();
  engine.start();

  const double runtime = timer.current_time();
  dc.cout()
    << "----------------------------------------------------------" << std::endl
    << "Final Runtime (seconds):   " << runtime
    << std::endl
    << "Updates executed: " << engine.num_updates() << std::endl
    << "Update Rate (updates/second): "
    << engine.num_updates() / runtime << std::endl;



  if(!word_dir.empty()) {
    // save word topic counts
    const bool gzip_output = false;
    const bool save_vertices = true;
    const bool save_edges = false;
    const size_t threads_per_machine = 2;
    const bool save_words = true;
    graph.save(word_dir, count_saver(save_words),
               gzip_output, save_vertices,
               save_edges, threads_per_machine);
  }


  if(!doc_dir.empty()) {
    // save doc topic counts
    const bool gzip_output = false;
    const bool save_vertices = true;
    const bool save_edges = false;
    const size_t threads_per_machine = 2;
    const bool save_words = false;
    graph.save(doc_dir, count_saver(save_words),
               gzip_output, save_vertices,
               save_edges, threads_per_machine);

  }


  graphlab::stop_metric_server_on_eof();
  graphlab::mpi_tools::finalize();
  return EXIT_SUCCESS;


} // end of main
