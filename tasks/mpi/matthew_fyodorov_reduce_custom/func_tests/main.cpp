#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/matthew_fyodorov_reduce_custom/include/ops_mpi.hpp"

TEST(matthew_fyodorov_reduce_custom_mpi, TestMPITaskParallel_Sum_PositiveNumbers) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (size > 1) {
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<int> output(1);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs_count.emplace_back(input.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));

    taskData->outputs_count.emplace_back(1);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

    matthew_fyodorov_reduce_custom_mpi ::TestMPITaskParallel task(taskData);
    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());

    if (rank == 0) {
      ASSERT_EQ(output[0], 15);
    }
  }
}

TEST(matthew_fyodorov_reduce_custom_mpi, TestMPITaskParallel_Sum_PositiveNumbers_equiualents_zero) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (size > 1) {
    std::vector<int> input = {1, -1, 2, -1, -1};
    std::vector<int> output(1);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs_count.emplace_back(input.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));

    taskData->outputs_count.emplace_back(1);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

    matthew_fyodorov_reduce_custom_mpi ::TestMPITaskParallel task(taskData);
    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());

    if (rank == 0) {
      ASSERT_EQ(output[0], 0);
    }
  }
}

TEST(matthew_fyodorov_reduce_custom_mpi, TestMPITaskParallel_Sum_PositiveNumbers_2) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (size > 1) {
    std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> output(1);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs_count.emplace_back(input.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));

    taskData->outputs_count.emplace_back(1);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

    matthew_fyodorov_reduce_custom_mpi ::TestMPITaskParallel task(taskData);
    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());

    if (rank == 0) {
      ASSERT_EQ(output[0], 55);
    }
  }
}

TEST(matthew_fyodorov_reduce_custom_mpi, TestMPITaskParallel_Sum_Positive_and_NegativeNumbers) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (size > 1) {
    std::vector<int> input = {1, -2, 3, -4, 5};
    std::vector<int> output(1);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs_count.emplace_back(input.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));

    taskData->outputs_count.emplace_back(1);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

    matthew_fyodorov_reduce_custom_mpi ::TestMPITaskParallel task(taskData);
    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());

    if (rank == 0) {
      ASSERT_EQ(output[0], 3);
    }
  }
}

TEST(matthew_fyodorov_reduce_custom_mpi, TestMPITaskParallel_Sum_Zero_Numbers) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (size > 1) {
    std::vector<int> input = {0};
    std::vector<int> output(1);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs_count.emplace_back(input.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));

    taskData->outputs_count.emplace_back(1);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

    matthew_fyodorov_reduce_custom_mpi ::TestMPITaskParallel task(taskData);
    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());

    if (rank == 0) {
      ASSERT_EQ(output[0], 0);
    }
  }
}

TEST(matthew_fyodorov_reduce_custom_mpi, TestMPITaskParallel_Sum_Random_Numbers) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (size > 1) {
    std::vector<int> input = matthew_fyodorov_reduce_custom_mpi ::getRandomVector(5);
    std::vector<int> output(1);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs_count.emplace_back(input.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));

    taskData->outputs_count.emplace_back(1);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

    matthew_fyodorov_reduce_custom_mpi ::TestMPITaskParallel task(taskData);
    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());

    if (rank == 0) {
      int expected_sum = std::accumulate(input.begin(), input.end(), 0);
      ASSERT_EQ(output[0], expected_sum);
    }
  }
}