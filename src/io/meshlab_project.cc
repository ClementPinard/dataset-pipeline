#include "io/meshlab_project.h"

#include <thirdparty/tinyxml2/tinyxml2.h>
#include <Eigen/Core>

using namespace tinyxml2;

namespace io {

bool ReadMeshLabProject(const std::string& project_file_path, MeshLabMeshInfoVector* meshes) {
  XMLDocument doc;
  if (doc.LoadFile(project_file_path.c_str()) != XML_SUCCESS) {
    return false;
  }
  
  XMLElement* xml_meshlabproject = doc.FirstChildElement("MeshLabProject");
  if (!xml_meshlabproject) {
    return false;
  }
  XMLElement* xml_mesh_group = xml_meshlabproject->FirstChildElement("MeshGroup");
  if (!xml_mesh_group) {
    return false;
  }
  XMLElement* xml_mlmesh = xml_mesh_group->FirstChildElement("MLMesh");
  while (xml_mlmesh) {
    MeshLabProjectMeshInfo new_mesh;
    if (xml_mlmesh->Attribute("label")) {
      new_mesh.label = xml_mlmesh->Attribute("label");
    }
    if (xml_mlmesh->Attribute("filename")) {
      new_mesh.filename = xml_mlmesh->Attribute("filename");
    }
    
    XMLElement* xml_mlmatrix44 = xml_mlmesh->FirstChildElement("MLMatrix44");
    if (xml_mlmatrix44) {
      std::string mlmatrix44_text = xml_mlmatrix44->GetText();
      std::istringstream mlmatrix44_stream(mlmatrix44_text);
      Eigen::Matrix4f M;
      mlmatrix44_stream >> M(0, 0) >> M(0, 1) >> M(0, 2) >> M(0, 3);
      mlmatrix44_stream >> M(1, 0) >> M(1, 1) >> M(1, 2) >> M(1, 3);
      mlmatrix44_stream >> M(2, 0) >> M(2, 1) >> M(2, 2) >> M(2, 3);
      mlmatrix44_stream >> M(3, 0) >> M(3, 1) >> M(3, 2) >> M(3, 3);
      new_mesh.global_T_mesh_full = M;
      new_mesh.global_T_mesh = Sophus::SE3f(M.block<3, 3>(0, 0), M.block<3, 1>(0, 3));
    } else {
      // Default-constructed transformation will be identity.
    }
    
    meshes->push_back(new_mesh);
    xml_mlmesh = xml_mlmesh->NextSiblingElement("MLMesh");
  }
  
  return true;
}

bool WriteMeshLabProject(const std::string& project_file_path, const MeshLabMeshInfoVector& meshes) {
  XMLDocument doc;

  XMLElement* xml_meshlabproject = doc.NewElement("MeshLabProject");
  doc.InsertEndChild(xml_meshlabproject);
  
  XMLElement* xml_meshgroup = doc.NewElement("MeshGroup");
  xml_meshlabproject->InsertEndChild(xml_meshgroup);
  
  for (const MeshLabProjectMeshInfo& mesh : meshes) {
    XMLElement* xml_mlmesh = doc.NewElement("MLMesh");
    xml_mlmesh->SetAttribute("label", mesh.label.c_str());
    xml_mlmesh->SetAttribute("filename", mesh.filename.c_str());
    xml_meshgroup->InsertEndChild(xml_mlmesh);
    
    XMLElement* xml_mlmatrix44 = doc.NewElement("MLMatrix44");
    std::ostringstream mlmatrix44_stream;
    mlmatrix44_stream << std::endl;
    // The spaces at the end are important. If omitted, MeshLab will crash when
    // opening the file.
    Eigen::Matrix3f R = mesh.global_T_mesh.so3().matrix();
    mlmatrix44_stream << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << mesh.global_T_mesh.translation()(0) << " " << std::endl;
    mlmatrix44_stream << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " " << mesh.global_T_mesh.translation()(1) << " " << std::endl;
    mlmatrix44_stream << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " " << mesh.global_T_mesh.translation()(2) << " " << std::endl;
    mlmatrix44_stream << "0 0 0 1 " << std::endl;
    xml_mlmatrix44->SetText(mlmatrix44_stream.str().c_str());
    xml_mlmesh->InsertEndChild(xml_mlmatrix44);
  }
  
  return (doc.SaveFile(project_file_path.c_str()) == tinyxml2::XML_NO_ERROR);
}

}  // namespace io
