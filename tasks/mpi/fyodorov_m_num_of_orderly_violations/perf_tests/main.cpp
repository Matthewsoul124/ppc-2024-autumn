// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "mpi/fyodorov_m_num_of_orderly_violations/include/ops_mpi.hpp"


TEST(mpi_fyodorov_m_num_of_orderly_violations_perf_test, test_large_input_pipeline_run) {
    boost::mpi::communicator world;
    std::vector<int> global_vec;
    std::vector<int32_t> global_sum(1, 0);
    
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    int count_size_vector;
    if (world.rank() == 0) {
        count_size_vector = 1000;  // Increased input size for stress testing
        global_vec = std::vector<int>(count_size_vector, 1);
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
        taskDataPar->inputs_count.emplace_back(global_vec.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    // Broadcast the vector size to all ranks
    boost::mpi::broadcast(world, count_size_vector, 0);
    if (world.rank() != 0) {
        global_vec.resize(count_size_vector, 1);
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
        taskDataPar->inputs_count.emplace_back(global_vec.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    auto testMpiTaskParallel = std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar, "+");
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    // Create Perf attributes
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    // Create and init perf results
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    
    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults);
        ASSERT_EQ(count_size_vector, global_sum[0]);
    }
}

TEST(mpi_fyodorov_m_num_of_orderly_violations_perf_test, test_large_input_task_run) {
    boost::mpi::communicator world;
    std::vector<int> global_vec;
    std::vector<int32_t> global_sum(1, 0);
    
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    int count_size_vector;
    if (world.rank() == 0) {
        count_size_vector = 10000;  // Increased input size for stress testing
        global_vec = std::vector<int>(count_size_vector, 1);
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
        taskDataPar->inputs_count.emplace_back(global_vec.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    // Broadcast the vector size to all ranks
    boost::mpi::broadcast(world, count_size_vector, 0);
    if (world.rank() != 0) {
        global_vec.resize(count_size_vector, 1);
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
        taskDataPar->inputs_count.emplace_back(global_vec.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }


    auto testMpiTaskParallel = std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar, "+");
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    // Create Perf attributes
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    // Create and init perf results
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->task_run(perfAttr, perfResults);
    
    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults);
        ASSERT_EQ(count_size_vector, global_sum[0]);
    }


}


