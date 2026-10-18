import sys, vtk
from vtk.util.numpy_support import vtk_to_numpy
 
src = sys.argv[1] if len(sys.argv) > 1 else "unscaled model/Geometry/ExoP.STL"
dst = sys.argv[2] if len(sys.argv) > 2 else "unscaled model/Geometry/ExoP.vtp"
target = int(sys.argv[3]) if len(sys.argv) > 3 else 10000   # max triangles to keep
r = vtk.vtkSTLReader(); r.SetFileName(src)
clean = vtk.vtkCleanPolyData(); clean.SetInputConnection(r.GetOutputPort())
tri = vtk.vtkTriangleFilter(); tri.SetInputConnection(clean.GetOutputPort())
tri.Update()
n_in = tri.GetOutput().GetNumberOfPolys()
last = tri
if n_in > target:   # decimate big CAD meshes so OpenSim can load them quickly
    dec = vtk.vtkQuadricDecimation(); dec.SetInputConnection(tri.GetOutputPort())
    dec.SetTargetReduction(1 - target / n_in); dec.Update(); last = dec
print(f"input triangles: {n_in}")
nrm = vtk.vtkPolyDataNormals(); nrm.SetInputConnection(last.GetOutputPort())
nrm.ConsistencyOn(); nrm.AutoOrientNormalsOn(); nrm.SplittingOff(); nrm.Update()
pd = nrm.GetOutput()
 
print(f"output: {pd.GetNumberOfPoints()} points, {pd.GetNumberOfPolys()} triangles")
if dst.lower().endswith(".stl"):          # binary STL: small and fast for OpenSim to load
    w = vtk.vtkSTLWriter(); w.SetFileName(dst); w.SetInputData(pd); w.SetFileTypeToBinary(); w.Write()
    print(f"wrote {dst}"); sys.exit()
 
pts = vtk_to_numpy(pd.GetPoints().GetData())
normals = vtk_to_numpy(pd.GetPointData().GetNormals())
polys = pd.GetPolys()
conn = vtk_to_numpy(polys.GetConnectivityArray())
offs = vtk_to_numpy(polys.GetOffsetsArray())[1:]   # VTK stores a leading 0; .vtp files don't
 
def rows(a): return "\n".join(" ".join(f"{v:.7g}" for v in row) for row in a)
def ints(a): return "\n".join(" ".join(str(int(v)) for v in a[i:i+12]) for i in range(0, len(a), 12))
 
with open(dst, "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n<PolyData>\n')
    f.write(f'<Piece NumberOfPoints="{len(pts)}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{len(offs)}">\n')
    f.write('<PointData Normals="Normals">\n<DataArray type="Float32" Name="Normals" NumberOfComponents="3" format="ascii">\n')
    f.write(rows(normals) + '\n</DataArray>\n</PointData>\n')
    f.write('<Points>\n<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
    f.write(rows(pts) + '\n</DataArray>\n</Points>\n')
    f.write('<Polys>\n<DataArray type="Int32" Name="connectivity" format="ascii">\n')
    f.write(ints(conn) + '\n</DataArray>\n<DataArray type="Int32" Name="offsets" format="ascii">\n')
    f.write(ints(offs) + '\n</DataArray>\n</Polys>\n</Piece>\n</PolyData>\n</VTKFile>\n')
print(f"wrote {dst}: {len(pts)} points, {len(offs)} triangles")
