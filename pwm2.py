 #Software PWM Servo2.py
import RPi.GPIO as GPIO
import time
P_SERVO = 13 # GPIO端口号，根据实际修改
fPWM = 50  # Hz (软件PWM方式，频率不能设置过高)
a = 10
b = 2
def setup():
    global pwm
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(P_SERVO, GPIO.OUT)
    pwm = GPIO.PWM(P_SERVO, fPWM)
    pwm.start(0)
def setDirect(direct):
    duty = a / 180 * direct + b
    pwm.ChangeDutyCycle(duty)
    print ("direct=", direct, "-> duty =", duty
)
    time.sleep(0.05) 
   
print ("starting"
)
setup()
for direct in range(90, 0, -30):
    setDirect(direct)
direct = 90  
setDirect(90)    
GPIO.cleanup() 
print ("done")