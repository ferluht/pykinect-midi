import sys
import argparse
from openni import openni2, nite2, utils
import numpy as np
import cv2
import PIL.Image, PIL.ImageTk
from math import sqrt

from tkinter import *

import time
import rtmidi

GRAY_COLOR = (64, 64, 64)
CAPTURE_SIZE_KINECT = (512, 424)
CAPTURE_SIZE_OTHERS = (640, 480)

def draw_limb(img, ut, j1, j2, col):
    (x1, y1) = ut.convert_joint_coordinates_to_depth(j1.position.x, j1.position.y, j1.position.z)
    (x2, y2) = ut.convert_joint_coordinates_to_depth(j2.position.x, j2.position.y, j2.position.z)

    x1 /= 2
    x2 /= 2
    y1 /= 2
    y2 /= 2

    if (0.4 < j1.positionConfidence and 0.4 < j2.positionConfidence):
        c = GRAY_COLOR if (j1.positionConfidence < 1.0 or j2.positionConfidence < 1.0) else col
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 1)

        c = GRAY_COLOR if (j1.positionConfidence < 1.0) else col
        cv2.circle(img, (int(x1), int(y1)), 2, c, -1)

        c = GRAY_COLOR if (j2.positionConfidence < 1.0) else col
        cv2.circle(img, (int(x2), int(y2)), 2, c, -1)


def draw_skeleton(img, ut, user, col):
    for idx1, idx2 in [(nite2.JointType.NITE_JOINT_HEAD, nite2.JointType.NITE_JOINT_NECK),
                       # upper body
                       (nite2.JointType.NITE_JOINT_NECK, nite2.JointType.NITE_JOINT_LEFT_SHOULDER),
                       (nite2.JointType.NITE_JOINT_LEFT_SHOULDER, nite2.JointType.NITE_JOINT_TORSO),
                       (nite2.JointType.NITE_JOINT_TORSO, nite2.JointType.NITE_JOINT_RIGHT_SHOULDER),
                       (nite2.JointType.NITE_JOINT_RIGHT_SHOULDER, nite2.JointType.NITE_JOINT_NECK),
                       # left hand
                       (nite2.JointType.NITE_JOINT_LEFT_HAND, nite2.JointType.NITE_JOINT_LEFT_ELBOW),
                       (nite2.JointType.NITE_JOINT_LEFT_ELBOW, nite2.JointType.NITE_JOINT_LEFT_SHOULDER),
                       # right hand
                       (nite2.JointType.NITE_JOINT_RIGHT_HAND, nite2.JointType.NITE_JOINT_RIGHT_ELBOW),
                       (nite2.JointType.NITE_JOINT_RIGHT_ELBOW, nite2.JointType.NITE_JOINT_RIGHT_SHOULDER),
                       # lower body
                       (nite2.JointType.NITE_JOINT_TORSO, nite2.JointType.NITE_JOINT_LEFT_HIP),
                       (nite2.JointType.NITE_JOINT_LEFT_HIP, nite2.JointType.NITE_JOINT_RIGHT_HIP),
                       (nite2.JointType.NITE_JOINT_RIGHT_HIP, nite2.JointType.NITE_JOINT_TORSO),
                       # left leg
                       (nite2.JointType.NITE_JOINT_LEFT_FOOT, nite2.JointType.NITE_JOINT_LEFT_KNEE),
                       (nite2.JointType.NITE_JOINT_LEFT_KNEE, nite2.JointType.NITE_JOINT_LEFT_HIP),
                       # right leg
                       (nite2.JointType.NITE_JOINT_RIGHT_FOOT, nite2.JointType.NITE_JOINT_RIGHT_KNEE),
                       (nite2.JointType.NITE_JOINT_RIGHT_KNEE, nite2.JointType.NITE_JOINT_RIGHT_HIP)]:
        draw_limb(img, ut, user.skeleton.joints[idx1], user.skeleton.joints[idx2], col)


class App:

    LEFT_HAND = 0
    LEFT_ELBOW = 1
    LEFT_SHOULDER = 2

    RIGHT_HAND = 3
    RIGHT_ELBOW = 4
    RIGHT_SHOULDER = 5

    TORSO = 6



    CLAP_DISTANCE = 100

    def __init__(self, window):
        self.window = window
        self.window.title("Kinect MIDI controller")
        # Create a canvas that can fit the above video source size

        self.init_camera()

        time.sleep(1)

        self.midiout = rtmidi.MidiOut()
        available_ports = self.midiout.get_ports()

        # here we're printing the ports to check that we see the one that loopMidi created.
        # In the list we should see a port called "loopMIDI port".
        print(available_ports)

        # Attempt to open the port
        for i, port in enumerate(available_ports):
            if ('kinect' in port):
                self.midiout.open_port(i)
                print('opened port ' + ' '.join(port.split()[:-1]))

        time.sleep(0.2)

        self.user_data = []

        for i in range(4):
            self.user_data.append({
                'positions': np.zeros((7, 3)),
                'gestures': {
                    'clap': False,
                    'strike': False,
                    'note_on': 0
                },
                'left_hand_to_shoulder_distance': 0
            })

        self.init_role_frame()
        self.init_param_frame()

        self.canvas = Canvas(window, width=self.win_w, height=self.win_h)
        self.canvas.grid(row=0, column=5, rowspan=10)

        self.window.wm_attributes("-topmost", 1)

        self.update()
        self.window.mainloop()

    def __del__(self):
        nite2.unload()
        openni2.unload()
        del self.midiout

    def send_CC(self, channel, cc, value):
        ctrl = [0xB0 | channel, cc, value]
        self.midiout.send_message(ctrl)

    def send_NoteON(self, channel, note, velocity):
        ctrl = [0x90 | channel, note, velocity]
        self.midiout.send_message(ctrl)

    def midi_ctrl(self, id, user):

        def store_position(nite_joint, local_joint):
            pos = user.skeleton.joints[nite_joint].position
            self.user_data[id]['positions'][local_joint][0] = pos.x
            self.user_data[id]['positions'][local_joint][1] = pos.y
            self.user_data[id]['positions'][local_joint][2] = pos.z

        store_position(nite2.JointType.NITE_JOINT_LEFT_HAND, self.LEFT_HAND)
        store_position(nite2.JointType.NITE_JOINT_LEFT_SHOULDER, self.LEFT_SHOULDER)
        store_position(nite2.JointType.NITE_JOINT_LEFT_ELBOW, self.LEFT_ELBOW)
        store_position(nite2.JointType.NITE_JOINT_RIGHT_HAND, self.RIGHT_HAND)
        store_position(nite2.JointType.NITE_JOINT_RIGHT_SHOULDER, self.RIGHT_SHOULDER)
        store_position(nite2.JointType.NITE_JOINT_RIGHT_ELBOW, self.RIGHT_ELBOW)
        store_position(nite2.JointType.NITE_JOINT_TORSO, self.TORSO)

        lhand = self.user_data[id]['positions'][self.LEFT_HAND]
        lshoulder = self.user_data[id]['positions'][self.LEFT_SHOULDER]
        lelbow = self.user_data[id]['positions'][self.LEFT_ELBOW]

        rhand = self.user_data[id]['positions'][self.RIGHT_HAND]
        rshoulder = self.user_data[id]['positions'][self.RIGHT_SHOULDER]
        relbow = self.user_data[id]['positions'][self.RIGHT_ELBOW]

        torso = self.user_data[id]['positions'][self.TORSO]

        hands_distance = np.linalg.norm(lhand - rhand)
        if (not self.user_data[id]['gestures']['clap']) and hands_distance < self.CLAP_DISTANCE:
            self.user_data[id]['gestures']['clap'] = True

            if ((lhand + rhand - lshoulder - rshoulder) / 2)[0] > 0:
                if ((lhand + rhand - lshoulder - rshoulder) / 2)[1] > 0:
                    self.send_CC(4, 81, 127)
                    self.focus_role = 1
                else:
                    self.send_CC(4, 83, 127)
                    self.focus_role = 3
            else:
                if ((lhand + rhand - lshoulder - rshoulder) / 2)[1] > 0:
                    self.send_CC(4, 80, 127)
                    self.focus_role = 0
                else:
                    self.send_CC(4, 82, 127)
                    self.focus_role = 2

        elif self.user_data[id]['gestures']['clap'] and hands_distance > self.CLAP_DISTANCE:
            self.user_data[id]['gestures']['clap'] = False

        def check_boundaries(val):
            val = int(val)
            if val > 127: val = 127
            if val < 0: val = 0
            return val

        larm_length = np.linalg.norm(lhand - lelbow) + np.linalg.norm(lelbow - lshoulder)
        rarm_length = np.linalg.norm(rhand - relbow) + np.linalg.norm(relbow - rshoulder)

        lhand_x = check_boundaries((lshoulder[0] - lhand[0]) / larm_length * 127)
        lhand_y = check_boundaries((lhand[1] - lshoulder[1] + larm_length/2) / larm_length * 127)
        lhand_z = check_boundaries((lshoulder[2] - lhand[2]) / larm_length * 127)
        rhand_x = check_boundaries((rhand[0] - rshoulder[0]) / rarm_length * 127)
        rhand_y = check_boundaries((rhand[1] - rshoulder[1] + rarm_length/2) / rarm_length * 127)
        rhand_z = check_boundaries((rshoulder[2] - rhand[2]) / rarm_length * 127)

        self.send_CC(self.focus_role, 1, lhand_x)
        self.send_CC(self.focus_role, 2, lhand_y)
        self.send_CC(self.focus_role, 3, lhand_z)

        self.send_CC(self.focus_role, 4, rhand_x)
        self.send_CC(self.focus_role, 5, rhand_y)
        self.send_CC(self.focus_role, 6, rhand_z)


    def init_role_frame(self):

        self.role_frame = Frame(self.window, bd=1)
        self.role_frame.grid(sticky=NW, row=0, column=0, columnspan=1)

        self.role_text = Label(self.role_frame, text='Roles', font='arial 12')
        self.role_text.grid(sticky=NW, row=0, column=0, columnspan=1)

        self.role_list = Listbox(self.role_frame, height=5, width=50, selectmode=SINGLE)
        self.roles = ["user1", "user2", "user3", "user4"]
        for role in self.roles:
            self.role_list.insert(END, role)
        self.role_list.grid(row=1, column=0, columnspan=3)
        self.role_list.bind("<<ListboxSelect>>", self.role_selection_callback)

        self.role_name_label = Label(self.role_frame, text='Role name', font='arial 10')
        self.role_name_label.grid(sticky=NW, row=2, column=0, columnspan=1)

        self.role_name_sv = StringVar()
        self.role_name_sv.trace_add("write", self.role_name_changed_callback)
        self.role_name = Entry(self.role_frame, font='arial 10', textvariable=self.role_name_sv)
        self.role_name.grid(sticky=NW, row=2, column=1, columnspan=1)

        self.btn_role_map = Button(self.role_frame, text="SEND MIDI", width=15, command=self.send_role_midi)
        self.btn_role_map.grid(sticky=NW, row=2, column=2, columnspan=1)

        self.focus_role = 0

    def send_role_midi(self):
        self.send_CC(4, self.focus_role + 80, 50)

    def init_param_frame(self):

        self.param_frame = Frame(self.window, bd=1)
        self.param_frame.grid(sticky=NW, row=1, column=0, columnspan=1)

        self.param_text = Label(self.param_frame, text='Parameters', font='arial 12')
        self.param_text.grid(sticky=NW, row=0, column=0, columnspan=1)

        self.param_list = Listbox(self.param_frame, height=5, width=50, selectmode=SINGLE)
        params = [     "left hand x", "left hand y", "left hand z",
                       "right hand x", "right hand y", "right hand z",
                       "head moving y",
                       "body moving z"]
        self.params = {}
        for param in params:
            self.param_list.insert(END, param)
            self.params[param] = {
                'on': 0,
            }
        self.param_list.grid(row=1, column=0, columnspan=3)
        self.param_list.bind("<<ListboxSelect>>", self.param_selection_callback)

        self.btn_map = Button(self.param_frame, text="SEND MIDI", width=15, command=self.send_midi)
        self.btn_map.grid(sticky=NW, row=2, column=1, columnspan=1)

        self.focus_param = 0

    def param_selection_callback(self, evt):
        # Note here that Tkinter passes an event object to onselect()
        w = evt.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        self.focus_param = index
        print ('You selected param %d: "%s"' % (index, value))

    def role_name_changed_callback(self, arg1, arg2, sv):
        self.role_list.delete(self.focus_role)
        self.role_list.insert(self.focus_role, self.role_name_sv.get())
        return True

    def role_selection_callback(self, evt):
        # Note here that Tkinter passes an event object to onselect()
        w = evt.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        self.focus_role = index
        self.role_name_sv.set(value)
        print ('You selected item %d: "%s"' % (index, value))

    def send_midi(self):
        self.send_CC(self.focus_role, self.focus_param+1, 50)

    def init_camera(self):

        openni2.initialize()
        nite2.initialize()
        self.dev = openni2.Device.open_any()

        dev_name = self.dev.get_device_info().name.decode('UTF-8')
        print("Device Name: {}".format(dev_name))
        self.use_kinect = False
        if dev_name == 'Kinect':
            self.use_kinect = True
            print('using Kinect.')

        try:
            self.user_tracker = nite2.UserTracker(self.dev)
        except utils.NiteError:
            print("Unable to start the NiTE human tracker. Check "
                  "the error messages in the console. Model data "
                  "(s.dat, h.dat...) might be inaccessible.")
            sys.exit(-1)

        (self.img_w, self.img_h) = CAPTURE_SIZE_KINECT if self.use_kinect else CAPTURE_SIZE_OTHERS
        self.win_w = 256
        self.win_h = int(self.img_h * self.win_w / self.img_w)

    def get_frame(self):
        ut_frame = self.user_tracker.read_frame()

        depth_frame = ut_frame.get_depth_frame()
        depth_frame_data = depth_frame.get_buffer_as_uint16()
        img = np.ndarray((depth_frame.height, depth_frame.width), dtype=np.uint16,
                         buffer=depth_frame_data).astype(np.float32)
        if self.use_kinect:
            img = img[0:self.img_h, 0:self.img_w]

        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(img)
        if (min_val < max_val):
            img = (img - min_val) / (max_val - min_val)

        self.frame = cv2.resize(img, (self.win_w, self.win_h))
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2RGB)

        if ut_frame.users:
            for user in ut_frame.users:
                if user.is_new():
                    print("new human id:{} detected.".format(user.id))
                    self.user_tracker.start_skeleton_tracking(user.id)
                elif (user.state == nite2.UserState.NITE_USER_STATE_VISIBLE and
                      user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED):
                    draw_skeleton(self.frame, self.user_tracker, user, (255, 0, 0))
                    self.midi_ctrl(user.id - 1, user)

        del img

    def update(self):
        self.get_frame()
        self.frame *= 255

        self.photo = PIL.ImageTk.PhotoImage(master=self.canvas, image=PIL.Image.fromarray(self.frame.astype('uint8')))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        del self.frame
        self.window.after(20, self.update)


if __name__ == '__main__':

    App(Tk())