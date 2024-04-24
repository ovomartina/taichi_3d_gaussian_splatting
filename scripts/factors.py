import symforce.symbolic as sf

def delta_factor_b_spline4_pixel(ut: sf.Scalar, x0: sf.Pose3, x1: sf.Pose3, x2: sf.Pose3, x3: sf.Pose3, x4: sf.Pose3,
                                y: sf.Pose3, sqrt_info: sf.M22, pixel:sf.V2, K:sf.Matrix33, epsilon: sf.Scalar) -> sf.V3:
    
    x = splines.b_spline4_value(ut, [x0, x1, x2, x3, x4], epsilon, None)
    
    T_w_c = x.to_homogenous_matrix()
    
    x_c = (pixel[0]-K[0,2]) / K[0,0]
    y_c = (pixel[1]-K[1,2]) / K[1,1]
    
    X_w = T_w_c @ sf.V3([x_c, y_c, 1.0])
    
    return delta_factor_r2(X_w[0:1], y.position()[0:1], sqrt_info, epsilon)