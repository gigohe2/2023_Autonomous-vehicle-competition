class Carstatus():
    def __init__(self, x=0, y=0, v=0, heading=0, speed=0, angle=0):
        self.x = x
        self.y = y
        self.v = v
        self.heading = heading
        self.speed = speed
        self.angle = angle

class PID():
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, dt=0.01, max_u = 1):
        self.Kp = Kp # 비례 gain
        self.Ki = Ki # 적분 gain
        self.Kd = Kd # 미분 gain
        self.dt = dt # 시간 간격
        self.max_u = max_u # 최대 제어 신호 크기. 시뮬레이터에서 최대 조향각

        self.i_err = 0 # error의 적분값
        self.prev_err = 0 # 이전의 error 값

    def do(self, err):
        up = self.Kp * err
        ui = self.Ki * self.i_err
        de = (err - self.prev_err) / self.dt
        self.prev_err = err
        ud = self.Kd * de

        u = up + ui + ud

        u = max(min(u, self.max_u), -self.max_u)
        self.i_err += err * self.dt 
        
        return u 