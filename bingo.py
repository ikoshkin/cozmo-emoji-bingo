import asyncio
import functools
import math
import numpy
import sys
import os
import shutil

import cozmo

from cozmo.util import degrees, distance_mm, radians, speed_mmps, Vector2
from cozmo.lights import Color, Light
from PIL import Image, ImageColor, ImageDraw, ImageStat

from classify import ImageClassifier

LOOK_AROUND_STATE = 'look_around'
PLAY_BINGO_STATE = 'play_bingo'
FOUND_EMOJI_STATE = 'found_emoji'
DRIVING_STATE = 'driving'
DONOTHING_STATE = 'nothin'

ANNOTATOR_WIDTH = 640.0
ANNOTATOR_HEIGHT = 480.0

DOWNSIZE_WIDTH = 32
DOWNSIZE_HEIGHT = 24


class EmojiBingoPlayer(cozmo.annotate.Annotator):
    '''
    Descitpiton

    Args:
        robot (cozmo.robot.Robot): instance of the robot connected from run_program.
    '''
    def __init__(self, robot: cozmo.robot.Robot):

        self.robot = robot
        self.robot.camera.image_stream_enabled = True
        # self.robot.camera.color_image_enabled = True

        self.robot.add_event_handler(cozmo.objects.EvtObjectTapped, self.on_cube_tap)
        self.robot.add_event_handler(cozmo.world.EvtNewCameraImage, self.on_new_camera_image)

        self.state = PLAY_BINGO_STATE
        self.emojician = ImageClassifier()
        self.last_detected_emoji = None

        self.game_selector_cube = None

        self.look_around_behavior = None  # type: LookAroundInPlace behavior
        self.drive_action = None  # type: DriveStraight action
        self.tilt_head_action = None  # type: SetHeadAngle action
        self.rotate_action = None  # type: TurnInPlace action
        self.lift_action = None  # type: SetLiftHeight action

    
    async def start_lookaround(self):
        '''Turns to a likely spot for a blob to be, then starts self.look_around_behavior.'''
        if self.look_around_behavior == None or not self.look_around_behavior.is_active:
            # self.turn_toward_last_known_blob()
            await asyncio.sleep(.5)
            if self.state == LOOK_AROUND_STATE:  # state may have changed due to turn_toward_last_known_blob
                self.abort_actions(self.tilt_head_action,self.rotate_action, self.drive_action)
                self.look_around_behavior = self.robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)

    def on_cube_tap(self, evt, obj, **kwargs):
        '''The blinking white cube switches the viewer between normal mode and pixel mode.
        The other illuminated cube toggles self.color_to_find.       
        '''
        if self.state == DONOTHING_STATE:
            self.state = PLAY_BINGO_STATE
            self.turn_on_cubes()

        



        # if obj.object_id == self.color_selector_cube.object_id:
        #     self.toggle_color_to_find()
        # elif obj.object_id == self.grid_cube.object_id:
        #     self.robot.world.image_annotator.annotation_enabled = not self.robot.world.image_annotator.annotation_enabled
        # elif obj.object_id == self.white_balance_cube.object_id:
        #     self.white_balance()
        
    def on_new_camera_image(self, evt, **kwargs):
        '''Processes the blobs in Cozmo's view, and determines the correct reaction.'''
        # downsized_image = self.get_low_res_view()
        # if ENABLE_COLOR_BALANCING:
        #     downsized_image = color_balance(downsized_image)
        if self.state == PLAY_BINGO_STATE:

            self.state = DONOTHING_STATE
            self.turn_off_cubes()

            PILimage = self.robot.world.latest_image.raw_image
            image_number = self.robot.world.latest_image.raw_image
            image_number = 'last'
            photo_location = f"photos/cozmo-{image_number}.jpeg"
            PILimage.save(photo_location, "JPEG")

            self.emojician.classify(photo_location)
            self.emojician.print_top_n()

            detected_emoji = self.emojician.top_n[0][0]
            prob = self.emojician.top_n[0][1]

            
            

            # detected_emoji, prob  = self.emojician.classify(photo_location)
            # print(detected_emoji)
            # print(prob)

            if prob > 0.75:
                # self.detected_emoji = detected_emoji
                # if self.state == PLAY_BINGO_STATE:
                #     self.detected_emoji = detected_emoji
                #     self.state = FOUND_EMOJI_STATE
                #     if self.look_around_behavior:
                #         self.look_around_behavior.stop()
                #         self.look_around_behavior = None
                self.on_finding_emoji(detected_emoji)
            else:
                self.robot.say_text(f"I'm not sure").wait_for_completed()
        
    def on_finding_emoji(self, detected_emoji):
        
        # if detected_emoji is not self.last_detected_emoji:
        self.last_detected_emoji = detected_emoji
        self.robot.say_text(f"{detected_emoji}").wait_for_completed()


    def abort_actions(self, *actions):
        '''Aborts the input actions if they are currently running.

        Args:
            *actions (list): the list of actions
        '''
        for action in actions:
            if action != None and action.is_running:
                action.abort()

    def turn_off_cubes(self):
        '''Illuminates the two cubes that control self.color_to_find and set the viewer display.'''
        self.game_selector_cube.set_lights_off()

    def turn_on_cubes(self):
        '''Illuminates the two cubes that control self.color_to_find and set the viewer display.'''
        self.game_selector_cube.set_lights(cozmo.lights.white_light)

    


    def cubes_connected(self):
        '''Returns true if Cozmo connects to both cubes successfully.'''
        self.game_selector_cube = self.robot.world.get_light_cube(cozmo.objects.LightCube1Id)
        return not (self.game_selector_cube is None)

    async def run(self):
        '''Program runs until typing CRTL+C into Terminal/Command Prompt, 
        or by closing the viewer window.
        '''
        if not self.cubes_connected():
            print('Cubes did not connect successfully - check that they are nearby. You may need to replace the batteries.')
            return

        self.turn_on_cubes()
        # await self.robot.drive_straight(distance_mm(100), speed_mmps(50), should_play_anim=False).wait_for_completed()

        # Updates self.state and resets self.amount_turned_recently every 1 second.
        while True:
            await asyncio.sleep(1)

           

            # if self.state == LOOK_AROUND_STATE:
            #     await self.start_lookaround()

            # if self.state == PLAY_BINGO_STATE:
            #     await self.start_lookaround()

            # if self.state == TAKE_PICTURES:
            #     await self.start_lookaround()

            # if self.state == FOUND_EMOJI_STATE:
            #     await self.on

            # if self.state == FOUND_EMOJI_STATE # and self.amount_turned_recently < self.moving_threshold:
            #     self.state = DRIVING_STATE


            # self.amount_turned_recently = radians(0)

#: entry point
async def cozmo_program(robot: cozmo.robot.Robot):

    if os.path.exists('photos'):
        shutil.rmtree('photos')
    if not os.path.exists('photos'):
        os.makedirs('photos')
    
    cozmo_player = EmojiBingoPlayer(robot)
    await cozmo_player.run()

cozmo.robot.Robot.drive_off_charger_on_connect = True
cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)

