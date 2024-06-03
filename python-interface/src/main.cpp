#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <airbot/command/command_base.hpp>
#include <airbot/modules/controller/fk.hpp>
#include <airbot/modules/controller/fk_analytic.hpp>
#include <airbot/modules/controller/id_chain_rne.hpp>
#include <airbot/modules/controller/ik.hpp>
#include <airbot/modules/controller/ik_chain.hpp>
#include <memory>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace arm;

std::unique_ptr<Robot> createAgent(
    AnalyticFKSolver* fksolver, ChainIKSolver* iksolver, ChainIDSolver* idsolver,
    const char* can_interface, double vel = M_PI,
    std::string end_mode = "newteacher", bool constraint = false,
    bool realtime = true, std::optional<uint16_t> camera_id = std::nullopt) {
  return std::make_unique<Robot>(
      std::unique_ptr<FKSolver>(fksolver), std::unique_ptr<IKSolver>(iksolver),
      std::unique_ptr<IDSolver>(idsolver), can_interface, vel, end_mode,
      constraint, realtime, camera_id);
}

PYBIND11_MODULE(airbot, m) {
  m.doc() = "airbot";

  py::class_<AnalyticFKSolver>(m, "AnalyticFKSolver")
      .def(py::init<const std::string&>());
  py::class_<ChainIKSolver>(m, "ChainIKSolver")
      .def(py::init<const std::string&>());
  py::class_<ChainIDSolver>(m, "ChainIDSolver")
      .def(py::init<const std::string, const std::string>());

  py::class_<Robot, std::unique_ptr<Robot>>(m, "Robot")
      .def("get_target_pose", &Robot::get_target_pose)
      .def("get_target_joint_q", &Robot::get_target_joint_q)
      .def("get_target_joint_v", &Robot::get_target_joint_v)
      .def("get_target_joint_t", &Robot::get_target_joint_t)
      .def("get_target_translation", &Robot::get_target_translation)
      .def("get_target_rotation", &Robot::get_target_rotation)
      .def("get_current_pose", &Robot::get_current_pose)
      .def("get_current_joint_q", &Robot::get_current_joint_q)
      .def("get_current_joint_v", &Robot::get_current_joint_v)
      .def("get_current_joint_t", &Robot::get_current_joint_t)
      .def("get_current_translation", &Robot::get_current_translation)
      .def("get_current_rotation", &Robot::get_current_rotation)
      .def("get_current_end", &Robot::get_current_end)
      .def("set_target_pose",
           py::overload_cast<const std::vector<std::vector<double>>&>(
               &Robot::set_target_pose))
      .def("set_target_pose", py::overload_cast<const std::vector<double>&,
                                                const std::vector<double>&>(
                                  &Robot::set_target_pose))
      .def("set_target_joint_q", &Robot::set_target_joint_q,
           py::arg("target_joint_q"), py::arg("spd") = py::none())
      .def("set_target_joint_v", &Robot::set_target_joint_v,
           py::arg("target_joint_v"))
      .def("set_target_joint_t", &Robot::set_target_joint_t,
           py::arg("target_joint_t"))
      .def("set_target_translation", &Robot::set_target_translation,
           py::arg("target_translation"))
      .def("set_target_end", &Robot::set_target_end,
          py::arg("target_end"))
      .def("set_target_rotation", &Robot::set_target_rotation,
           py::arg("target_rotation"))
      .def("add_target_joint_q", &Robot::add_target_joint_q,
           py::arg("target_d_joint_q"))
      .def("add_target_joint_v", &Robot::add_target_joint_v,
           py::arg("target_d_joint_v"))
      .def("add_target_translation", &Robot::add_target_translation,
           py::arg("target_d_translation"))
      .def("add_target_relative_translation",
           &Robot::add_target_relative_translation,
           py::arg("target_d_translation"))
      .def("add_target_relative_rotation", &Robot::add_target_relative_rotation,
           py::arg("target_d_rotation"))
      .def("record_start", &Robot::record_start)
      .def("record_stop", &Robot::record_stop)
      .def("record_replay", &Robot::record_replay)
      .def("gravity_compensation", &Robot::gravity_compensation);

  m.def("create_agent", &createAgent, py::arg("fksolver"), py::arg("iksolver"),
        py::arg("idsolver"), py::arg("can_interface"), py::arg("vel") = M_PI,
        py::arg("end_mode") = "newteacher", py::arg("constraint") = false,
        py::arg("realtime") = true, py::arg("camera_id") = py::none());
};
