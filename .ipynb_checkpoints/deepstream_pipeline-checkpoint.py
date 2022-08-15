#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys

sys.path.append("../")
sys.path.append("./")
from common.bus_call import bus_call
from common.is_aarch_64 import is_aarch64
#from common.FPS_2 import PERF_DATA
from common.FPS import GETFPS

import pyds
import platform
import math
import time
from ctypes import *
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib
import configparser
import argparse


#perf_data = None
fps_streams={}

MAX_DISPLAY_LEN = 64
# PGIE_CLASS_ID_PERSON = 0
# PGIE_CLASS_ID_VEHICLE = 2
# PGIE_CLASS_ID_BUS = 5
# PGIE_CLASS_ID_TRUCK = 7
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_TRUCK = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_BUS = 3
MUXER_OUTPUT_WIDTH = 1280
MUXER_OUTPUT_HEIGHT = 960
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 960
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 0
past_tracking_meta=[0]

# tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.
def tiler_src_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    # obj_counter = {
    #     PGIE_CLASS_ID_PERSON:0,
    #     PGIE_CLASS_ID_VEHICLE:0,
    #     PGIE_CLASS_ID_BUS:0,
    #     PGIE_CLASS_ID_TRUCK:0
    # }
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_TRUCK:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BUS:0
    }
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        while l_obj is not None:
            try: 
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
                
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        #py_nvosd_text_params.display_text = "Frame Number={}".format(frame_number)
        #py_nvosd_text_params.display_text = "Frame Number={} Person_count={}".format(frame_number, obj_counter[PGIE_CLASS_ID_PERSON])
        py_nvosd_text_params.display_text = "Frame Number={} Car_count={} Bus_count={} Truck_count={} Pedestrian_count={}".format(frame_number, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_BUS], obj_counter[PGIE_CLASS_ID_TRUCK], obj_counter[PGIE_CLASS_ID_PERSON])
        py_nvosd_text_params.x_offset = 10;
        py_nvosd_text_params.y_offset = 12;
        
        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 12
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        
        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        
        # send the display overlay to the screen
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # print("Frame Number={} Number of Objects={}".format(frame_number, num_rects))

        # Get frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        fps_streams[stream_index].get_fps()
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
            
    #past traking meta data
    past_tracking = None
    past_tracking_meta[0] = past_tracking
    if(past_tracking_meta[0]==1):
        l_user=batch_meta.batch_user_meta_list
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting is done by pyds.NvDsUserMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone
                user_meta=pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
                try:
                    # Note that user_meta.user_meta_data needs a cast to pyds.NvDsPastFrameObjBatch
                    # The casting is done by pyds.NvDsPastFrameObjBatch.cast()
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone
                    pPastFrameObjBatch = pyds.NvDsPastFrameObjBatch.cast(user_meta.user_meta_data)
                except StopIteration:
                    break
                for trackobj in pyds.NvDsPastFrameObjBatch.list(pPastFrameObjBatch):
                    print("streamId=",trackobj.streamID)
                    print("surfaceStreamID=",trackobj.surfaceStreamID)
                    for pastframeobj in pyds.NvDsPastFrameObjStream.list(trackobj):
                        print("numobj=",pastframeobj.numObj)
                        print("uniqueId=",pastframeobj.uniqueId)
                        print("classId=",pastframeobj.classId)
                        print("objLabel=",pastframeobj.objLabel)
                        for objlist in pyds.NvDsPastFrameObjList.list(pastframeobj):
                            print('frameNum:', objlist.frameNum)
                            print('tBbox.left:', objlist.tBbox.left)
                            print('tBbox.width:', objlist.tBbox.width)
                            print('tBbox.top:', objlist.tBbox.top)
                            print('tBbox.right:', objlist.tBbox.height)
                            print('confidence:', objlist.confidence)
                            print('age:', objlist.age)
            try:
                l_user=l_user.next
            except StopIteration:
                break
                
    # Get meta data from NvDsAnalyticsFrameMeta
    l_user = frame_meta.frame_user_meta_list #Get glist containing NvDsUserMeta objects from given NvDsFrameMeta
    while l_user:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data) #Must cast glist data to NvDsUserMeta object
            if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSFRAME.USER_META"):
                user_meta_data = pyds.NvDsAnalyticsFrameMeta.cast(user_meta.user_meta_data) #Must cast user metadata to NvDsAnalyticsFrameMeta
                #Access NvDsAnalyticsFrameMeta attributes with user_meta_data.{attribute name}
                if user_meta_data.objInROIcnt: print("Objs in ROI: {0}".format(user_meta_data.objInROIcnt))                    
                if user_meta_data.objLCCumCnt: print("Linecrossing Cumulative: {0}".format(user_meta_data.objLCCumCnt))
                if user_meta_data.objLCCurrCnt: print("Linecrossing Current Frame: {0}".format(user_meta_data.objLCCurrCnt))
                if user_meta_data.ocStatus: print("Overcrowding status: {0}".format(user_meta_data.ocStatus))
        except StopIteration:
            break
        try:
            l_user = l_user.next
        except StopIteration:
            break
                
    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if "video" in gstname:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = f"source-bin-{index:02}"
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args):
    # Check input arguments
    #global perf_data
    #perf_data = PERF_DATA(len(args))
    for i in range(0,len(args)):
        fps_streams["stream{0}".format(i)]=GETFPS(i)
        
    number_sources = len(args)
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = f"sink_{i}"
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    preprocess = Gst.ElementFactory.make("nvdspreprocess", "preprocess-plugin")
    if not preprocess:
        sys.stderr.write(" Unable to create preprocess \n")
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating tracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")   
    print("Creating analytics \n ")
    analytics = Gst.ElementFactory.make("nvdsanalytics", "nvdsanalytics")
    if not analytics:
        sys.stderr.write(" Unable to create analytics \n")
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    )

    # Make the encoder
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property("bitrate", bitrate)
    if is_aarch64():
        encoder.set_property("preset-level", 1)
        encoder.set_property("insert-sps-pps", 1)
        #encoder.set_property("bufapi-version", 1)
        
    # added for mp4-out
    codecparse = Gst.ElementFactory.make("h264parse", "h264_parse")
    if not codecparse:
        sys.stderr.write(" Unable to create codecparse \n")
        
    mux = Gst.ElementFactory.make("mp4mux", "mux")
    if not mux:
        sys.stderr.write(" Unable to create mux \n")

    sink = Gst.ElementFactory.make("filesink", "filesink")
    if not sink:
        sys.stderr.write(" Unable to create filesink \n")
    sink.set_property('location', output_path)

#     # Make the payload-encode video into RTP packets
#     if codec == "H264":
#         rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
#         print("Creating H264 rtppay")
#     elif codec == "H265":
#         rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
#         print("Creating H265 rtppay")
#     if not rtppay:
#         sys.stderr.write(" Unable to create rtppay")

#     # Make the UDP sink
#     updsink_port_num = 5400
#     sink = Gst.ElementFactory.make("udpsink", "udpsink")
#     if not sink:
#         sys.stderr.write(" Unable to create udpsink")

    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    queue8=Gst.ElementFactory.make("queue","queue8")
    #queue9=Gst.ElementFactory.make("queue","queue9")
    #queue10=Gst.ElementFactory.make("queue","queue10")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    pipeline.add(queue8)
    #pipeline.add(queue9)
    #pipeline.add(queue10)

    # sink.set_property("host", "224.224.255.255")
    # sink.set_property("port", updsink_port_num)
    # sink.set_property("async", False)
    # sink.set_property("sync", 1)

    streammux.set_property("width", 1280)
    streammux.set_property("height", 960)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)
    preprocess.set_property("config-file", "config_preprocess.txt")
    analytics.set_property('config-file', "config_nvdsanalytics.txt")
    pgie.set_property("config-file-path", "config_infer_primary_yolo.txt")
    
    pgie_batch_size = pgie.get_property("batch-size")
    pgie.set_property("input-tensor-meta", True)
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)

    print("Adding elements to Pipeline \n")
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos", 0)
    
    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('config_tracker.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)
            
    print("Adding elements to Pipeline \n")        
    pipeline.add(preprocess)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(analytics)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(codecparse)
    pipeline.add(mux)
    # pipeline.add(rtppay)
    pipeline.add(sink)

    streammux.link(queue1)
    queue1.link(preprocess)
    preprocess.link(queue2)
    queue2.link(pgie)
    #pgie.link(queue3)
    pgie.link(tracker)
    tracker.link(analytics)
    analytics.link(queue3)
    queue3.link(tiler)
    tiler.link(queue4)
    queue4.link(nvvidconv)
    nvvidconv.link(queue5)
    queue5.link(nvosd)
    nvosd.link(queue6)
    queue6.link(nvvidconv_postosd)
    nvvidconv_postosd.link(queue7)
    queue7.link(caps)
    caps.link(queue8)
    queue8.link(encoder)
    encoder.link(codecparse)
    codecparse.link(mux)
    mux.link(sink)
    # encoder.link(queue9)
    # queue9.link(rtppay)
    # rtppay.link(queue10)
    # queue10.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    tiler_src_pad=pgie.get_static_pad("src")
    if not tiler_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0)

#     # Start streaming
#     rtsp_port_num = 8554

#     server = GstRtspServer.RTSPServer.new()
#     server.props.service = str(rtsp_port_num)
#     server.attach(None)

#     factory = GstRtspServer.RTSPMediaFactory.new()
#     factory.set_launch(
#         f'( udpsrc name=pay0 port={updsink_port_num} buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string){codec}, payload=96 " )'
#     )
#     factory.set_shared(True)
#     server.get_mount_points().add_factory("/ds-test", factory)

#     pgie_src_pad = pgie.get_static_pad("src")
#     if not pgie_src_pad:
#         sys.stderr.write(" Unable to get src pad \n")
#     else:
#         pgie_src_pad.add_probe(
#             Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0
#         )
#         # perf callback function to print fps every 5 sec
#         GLib.timeout_add(5000, perf_data.perf_print_callback)

#     print(f"\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:{rtsp_port_num}/ds-test ***\n\n")

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except BaseException:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description="RTSP Output Sample Application Help ")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input H264 elementry stream",
        nargs="+",
        default=["a"],
        required=True,
    )
    parser.add_argument(
        "-o", 
        "--output",
        default='./out.mp4',
        help="Set the output file path "
    )
    parser.add_argument(
        "-c",
        "--codec",
        default="H264",
        help="RTSP Streaming Codec H264/H265 , default=H264",
        choices=["H264", "H265"],
    )
    parser.add_argument(
        "-b", "--bitrate", default=4000000, help="Set the encoding bitrate ", type=int
    )
    # Check input arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global codec
    global output_path
    global bitrate
    global stream_path
    codec = args.codec
    bitrate = args.bitrate
    output_path = args.output
    stream_path = args.input
    return stream_path


if __name__ == "__main__":
    stream_path = parse_args()
    sys.exit(main(stream_path))
