import matplotlib.pyplot as plt
import menpo.io as mio
import menpo.shape
import numpy as np
import menpofit.modelinstance
from menpo.visualize import print_progress
# Import shape and reconstruct
#shape = mio.import_builtin_asset.einstein_pts().lms
#shape_model.set_target(shape)

# Visualize
#plt.subplot(121)
#shape.view(render_axes=False, axes_x_limits=0.05, axes_y_limits=0.05)
#plt.gca().set_title('Original shape')
#plt.subplot(122)
#shape_model.target.view(render_axes=False, axes_x_limits=0.05, axes_y_limits=0.05)
#plt.gca().set_title('Reconstructed shape');

ar = np.ndarray(shape=(10,2), dtype=float)
print(ar)
pc = menpo.shape.PointCloud(ar)

print("Dimensions " + str(pc.n_dims))
print("Points count " + str(pc.n_points))
print("Parameters " + str(pc.n_parameters))
print("Has landmarks " + str(pc.has_landmarks))

for lt in print_progress(pc):
    print(lt)
    
opdm = menpofit.modelinstance.OrthoPDM(pc)
print(opdm.n_points)
print(opdm.n_active_components)