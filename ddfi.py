import os,sys
import argparse
import subprocess

class args:
    pass
parser = argparse.ArgumentParser(description='animation auto duplicated frame remove and frame interpolate tool',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i','--input',required=True,type=str,help='source file, any format ffmpeg can decode\n ')
parser.add_argument('-o','--output',required=False,type=str,help='output file, default \"input file\"_interp.mkv\n ')
parser.add_argument('-tf','--temp_folder',required=False,type=str,help='temp folder, default \"output file\"_tmp\\\n ')
parser.add_argument('-st','--start_time',required=False,type=str,help='cut input video from this time, format h:mm:ss.nnn\n ')
parser.add_argument('-et','--end_time',required=False,type=str,help='cut input video end to this time, format h:mm:ss.nnn\n ')
parser.add_argument('-as','--audio_stream',required=True,type=int,help='set audio stream index, starts from 0, \"no\" means don\'t output audio\n ')
parser.add_argument('-q','--output_crf',required=False,type=int,help='output video crf value, interger. if a codec don\'t has -crf option is used, \n--ffmpeg_params_output can be used as a workaround. default 18\n ')
parser.add_argument('-qi','--intermedia_crf',required=False,type=int,help='intermedia video crf value, interger. default 12\n ')
parser.add_argument('-if','--interpolation_factor',required=False,type=int,help='interpolation factor, interger. default 8, \nnot recommend to decrease it\n ')
parser.add_argument('-tr','--target_framerate',required=False,type=str,help='target framerate, only affects final output, \ncan be fraction, interger or decimal. default 60000/1001\n ')
parser.add_argument('-vc','--video_codec',required=False,type=str,help='output video codec, use the name in ffmpeg, \nwill be constrained by output format. default libx264\n ')
parser.add_argument('-ac','--audio_codec',required=False,type=str,help='output audio codec, similar to -vc. default aac\n ')
parser.add_argument('-al','--audio_channel_layout',required=False,type=str,help='output audio channel layout, \nuse the name in ffmpeg. default stereo\n ')
parser.add_argument('-ab','--audio_bitrate',required=False,type=str,help='output audio bitrate, use it like in ffmpeg. default 256k\n ')
parser.add_argument('-cd','--consecutive_drop_max',required=False,type=int,help='set the maximum number of consecutive dropped frames. \ndefault 2. this will be passed to the \"max\" option of ffmpeg filter mpdecimate.\n ')
parser.add_argument('--ffmpeg_params_output',required=False,type=str,help='other ffmpeg parameters used in final step, \nuse it carefully. default \"-map_metadata -1\"\n ')
parser.parse_args(sys.argv[1:],args)

inFile=args.input
outFile=os.path.splitext(inFile)[0]+'_interp.mkv' if args.output is None else args.output
tmpFolder=os.path.splitext(outFile)[0]+'_tmp\\' if args.temp_folder is None else args.temp_folder+'\\'
#tmpFolder+='\\' if tmpFolder[-1] is not '\\' else ''

ffss='' if args.start_time is None else f'-ss {args.start_time}'
ffto='' if args.end_time is None else f'-to {args.end_time}'
ffau='' if args.audio_stream == 'no' else f'-map a:{args.audio_stream}'
ffau2='' if args.audio_stream == 'no' else f'-map 1:a:0'
crfo=18 if args.output_crf is None else args.output_crf
crfi=12 if args.intermedia_crf is None else args.intermedia_crf
xinterp=8 if args.interpolation_factor is None else args.interpolation_factor
fro='60000/1001' if args.target_framerate is None else args.target_framerate
codecov='-c:v libx264' if args.video_codec is None else f'-c:v {args.video_codec}'
codecoa='-c:a aac' if args.audio_codec is None else f'-c:a {args.audio_codec}'
clo='-channel_layout stereo' if args.audio_channel_layout is None else f'-channel_layout {args.audio_channel_layout}'
abo='-b:a 256k' if args.audio_bitrate is None else f'-b:a {args.audio_bitrate}'
ffparamo='-map_metadata -1' if args.ffmpeg_params_output is None else args.ffmpeg_params_output
mpdmax=2 if args.consecutive_drop_max is None else args.consecutive_drop_max

tmpDedup=os.path.abspath(tmpFolder+'dedup_done.mkv')
tmpInterpXn=f'{tmpFolder}interpX{xinterp}_done.264'
tmpTSV2O=f'{tmpFolder}tsv2o.txt'
tmpTSV2N=f'{tmpFolder}tsv2nX{xinterp}.txt'

ffpath=''
mmgpath='D:\\Softwares\\Mkvtoolnix\\'
dllpath=os.path.dirname(os.path.realpath(__file__))
vspipepath=''

if not os.path.exists(inFile):
    print('input file isn\'t exist')
    exit()
if os.path.exists(outFile):
    if input('output file exists, continue? (y/n) ')=='y':
        pass
    else:
        exit()
if os.path.exists(tmpFolder):
    if input('temp folder exists, continue? (y/n) ')=='y':
        pass
    else:
        exit()
else:
    os.mkdir(tmpFolder)

if not os.path.exists(tmpDedup):
    ff_mpdecimate=f'\"{ffpath}ffmpeg.exe\" {ffss} {ffto} -i \"{inFile}\" -vf mpdecimate=max={mpdmax} -crf {crfi} -preset 1 -pix_fmt yuv420p -sample_fmt s16 -c:v libx264 -c:a flac -map v:0 {clo} {ffau} \"{tmpFolder}dedup.mkv\" -y'
    print(ff_mpdecimate)
    subprocess.run(ff_mpdecimate,shell=True)
    os.rename(f'{tmpFolder}dedup.mkv',tmpDedup)

if not os.path.exists(tmpInterpXn):
    script='''import vapoursynth as vs
from vapoursynth import core
'''
    if not (os.path.exists(r"C:\Program Files\Vapoursynth\plugins\LSMASHSource.dll") or os.path.exists(r"C:\Program Files (x86)\Vapoursynth\plugins\LSMASHSource.dll")):
        script+=f'core.std.LoadPlugin(r\"{dllpath}\\LSMASHSource.dll\")\n'

    if not (os.path.exists(r"C:\Program Files\Vapoursynth\plugins\svpflow1_vs64.dll") or os.path.exists(r"C:\Program Files (x86)\Vapoursynth\plugins\svpflow1_vs64.dll")):
        script+=f'core.std.LoadPlugin(r\"{dllpath}\\svpflow1_vs64.dll\")\n'

    if not (os.path.exists(r"C:\Program Files\Vapoursynth\plugins\svpflow2_vs64.dll") or os.path.exists(r"C:\Program Files (x86)\Vapoursynth\plugins\svpflow2_vs64.dll")):
        script+=f'core.std.LoadPlugin(r\"{dllpath}\\svpflow2_vs64.dll\")\n'

    script+=f'clip = core.lsmas.LWLibavSource(r\"{tmpDedup}\")\nclip = core.std.AssumeFPS(clip,fpsnum=10,fpsden=1)\n'

    script+='''crop_string = ""
resize_string = ""
super_params = "{pel:2,gpu:1}"
analyse_params = "{block:{w:16,h:16},main:{search:{coarse:{distance:-10}}},refine:[{thsad:200}]}"
smoothfps_params = "{rate:{num:%d,den:2},algo:23,cubic:1}"
def interpolate(clip):
    input = clip
    if crop_string!='':
        input = eval(crop_string)
    if resize_string!='':
        input = eval(resize_string)
    super   = core.svp1.Super(input,super_params)
    vectors = core.svp1.Analyse(super["clip"],super["data"],input,analyse_params)
    smooth  = core.svp2.SmoothFps(input,super["clip"],super["data"],vectors["clip"],vectors["data"],smoothfps_params,src=clip)
    smooth  = core.std.AssumeFPS(smooth,fpsnum=smooth.fps_num,fpsden=smooth.fps_den)
    return smooth
clip =  interpolate(clip)
clip.set_output()
''' % (2*xinterp)

    vpy=open(f'{tmpFolder}interpX{xinterp}.vpy','w')
    print(script,file=vpy)
    vpy.close()
    ff_interp=f'\"{vspipepath}vspipe.exe\" -y \"{tmpFolder}interpX{xinterp}.vpy\" - | \"{ffpath}ffmpeg.exe\" -i - -crf {crfi} -preset 1 -pix_fmt yuv420p -c:v libx264 -map v:0 \"{tmpFolder}interp.264\" -y'
    print(ff_interp)
    subprocess.run(ff_interp,shell=True)
    os.rename(f'{tmpFolder}interp.264',tmpInterpXn)

if not os.path.exists(f'{tmpInterpXn}.mkv'):
    mEx=f'\"{mmgpath}mkvextract.exe\" \"{tmpDedup}\" timestamps_v2 0:\"{tmpTSV2O}\"'
    print(mEx)
    subprocess.run(mEx,shell=True)
    
    ts_o=list()
    ts_new=list()
    outfile=open(tmpTSV2N,'w',encoding='utf-8')
    f=open(tmpTSV2O,'r',encoding='utf-8')
    for line in f:
        ts_o.append(line)
    #print(ts_o)
    
    for x in range(1,len(ts_o)-1):
        ts_new.append(str(float(ts_o[x])))
        for i in range(1,xinterp):
            ts_new.append( str(float(ts_o[x]) + (float(ts_o[x+1])-float(ts_o[x]))/xinterp*i) )
    #print(ts_new)
    print('# timestamp format v2',file=outfile)
    for x in range(len(ts_new)):
        print(ts_new[x],file=outfile)
    print(ts_o[len(ts_o)-1],file=outfile)
    f.close()
    outfile.close()
    
    mmg=f'\"{mmgpath}mkvmerge.exe\" -o \"{tmpInterpXn}.mkv\" --timestamps 0:\"{tmpTSV2N}\" \"{tmpInterpXn}\"'
    print(mmg)
    subprocess.run(mmg,shell=True)

ff_down60=f'\"{ffpath}ffmpeg.exe\" -i \"{tmpInterpXn}.mkv\" -i {tmpDedup} -map 0:v:0 {ffau2} -vf framerate={fro} -crf {crfo} {codecov} {codecoa} {abo} {ffparamo} \"{outFile}\" -y'
print(ff_down60)
subprocess.run(ff_down60,shell=True)
