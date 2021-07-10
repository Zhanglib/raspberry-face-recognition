# Software PWM Servo.py
import RPi.GPIO as GPIO
import time
P_SERVO = 13 # GPIO端口号，根据实际修改
fPWM = 50  # Hz (软件PWM方式，频率不能设置过高)
a = 10
b = 2.5
def setup():
    global pwm
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(P_SERVO, GPIO.OUT)
    pwm = GPIO.PWM(P_SERVO, fPWM)
    pwm.start(0)
    #time.sleep(0.1) 
    
def setDirection(direction):
    duty = a / 180 * direction + b
    pwm.ChangeDutyCycle(duty)
    print ("direction =", direction, "-> duty =", duty
)
    time.sleep(0.02) 
   
print ("starting"
)
setup()
#pwm.start(0)
for direction in range(0, 45, 10):
    setDirection(direction)
direction = 0
for direction in range(45, -1, -10):
    setDirection(direction)
#setDirection(0)
pwm.stop()
GPIO.cleanup()
time.sleep(3) 
print ("done")