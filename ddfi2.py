import os,sys
import argparse
import subprocess
try:
    import psutil as mt
except ModuleNotFoundError:
    import multiprocessing as mt
toolsFolder=f'{os.path.dirname(os.path.realpath(__file__))}\\tools\\'
sys.path.append(toolsFolder)
class args:
    pass
parser = argparse.ArgumentParser(description='an animation auto duplicated frame remove and frame interpolate script, uses ffmpeg, mkvextract, and vapoursynth.',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i','--input',required=True,type=str,help='source file, any format ffmpeg can decode\n ')
parser.add_argument('-o','--output',required=False,type=str,help='output file, default \"input file\"_interp.mkv\n ')
parser.add_argument('-tf','--temp_folder',required=False,type=str,help='temp folder, default \"output file\"_tmp\\\n ')
parser.add_argument('-st','--start_time',required=False,type=str,help='cut input video start from this time, format h:mm:ss.nnn or seconds\n ')
parser.add_argument('-et','--end_time',required=False,type=str,help='cut input video end at this time, format h:mm:ss.nnn or seconds\n ')
parser.add_argument('-as','--audio_stream',required=False,type=str,help='set audio stream index, starts from 0, \n\"no\" means don\'t output audio, and is the default\n ',default='no')
parser.add_argument('-q','--output_crf',required=False,type=int,help='output video crf value, interger. if a codec don\'t has -crf option is used, \n--ffmpeg_params_output can be used as a workaround. default 18\n ')
parser.add_argument('-vc','--video_codec',required=False,type=str,help='output video codec, use the name in ffmpeg, \nwill be constrained by output format. default libx265\n ')
parser.add_argument('-ac','--audio_codec',required=False,type=str,help='output audio codec, similar to -vc. default libopus\n ')
parser.add_argument('-al','--audio_channel_layout',required=False,type=str,help='output audio channel layout, \nuse the name in ffmpeg. default stereo\n ')
parser.add_argument('-ab','--audio_bitrate',required=False,type=str,help='output audio bitrate, use it like in ffmpeg. default 128k\n ')
parser.add_argument('-ddt','--dedup_thresholds',required=False,type=str,help='ssim, max pixel diff (16 bits scale) and max consecutive deletion, inclusive. \ndefault 0.999,10240,2\n ',default='0.999,10240,2')
parser.add_argument('--ffmpeg_params_output',required=False,type=str,help='other ffmpeg parameters used in final step, \nuse it carefully. default \"-map_metadata -1 -map_chapters -1\"\n ')
parser.add_argument('-scd',required=False,type=str,help='scene change detect method, \"misc\", \"mv\" or \"none\", default mv\n ',default='mv')
parser.add_argument('-thscd',required=False,type=str,help='thscd1&2 of core.mv.SCDetection, default 200,85\n ',default='200,85')
parser.add_argument('-threads',required=False,type=int,help='how many threads to use in VS (core.num_threads), default auto detect\n ',default=None)
parser.add_argument('-maxmem',required=False,type=int,help='max memory to use for cache in VS (core.max_cache_size) in MB, default 4096\n ',default=4096)
parser.add_argument('-model',required=False,type=float,help='model version, default 4.0\n ',default=4.0)
parser.add_argument('--slower-model',required=False,action=argparse.BooleanOptionalAction,default=False)
parser.add_argument('-mf',required=False,type=str,help='medium fps.\n ',default="192000,1001")
parser.parse_args(sys.argv[1:],args)


inFile=args.input
outFile=os.path.splitext(inFile)[0]+'_interp.mkv' if args.output is None else args.output
tmpFolder=os.path.splitext(outFile)[0]+'_tmp\\' if args.temp_folder is None else args.temp_folder+'\\'
#tmpFolder+='\\' if tmpFolder[-1] is not '\\' else ''

ffss='' if args.start_time is None else f'-ss {args.start_time}'
ffto='' if args.end_time is None else f'-to {args.end_time}'
ffau='-an' if args.audio_stream == 'no' else f'-map a:{args.audio_stream}'
ffau2='-an' if args.audio_stream == 'no' else f'-map 1:a:0'
crfo=18 if args.output_crf is None else args.output_crf
x265_default='-c:v libx265 -preset 6 -x265-params sao=0:rect=0:strong-intra-smoothing=0:open-gop=0:b-intra=1:weightb=1:aq-mode=1:aq-strength=0.8:ctu=32:rc-lookahead=60:me=hex:subme=2'
codecov=x265_default if args.video_codec is None else x265_default+':'+args.video_codec[1:] if args.video_codec[0]=='+' else f'-c:v {args.video_codec}'
codecoa='-c:a libopus' if args.audio_codec is None else f'-c:a {args.audio_codec}'
clo='-channel_layout stereo' if args.audio_channel_layout is None else f'-channel_layout {args.audio_channel_layout}'
abo='-b:a 128k' if args.audio_bitrate is None else f'-b:a {args.audio_bitrate}'
ffparamo='-map_metadata -1 -map_chapters -1' if args.ffmpeg_params_output is None else args.ffmpeg_params_output
threads=args.threads if args.threads else int(mt.cpu_count()/2)
ddt=args.dedup_thresholds.split(',')
ssimt=float(ddt[0])
pxdifft=int(ddt[1])
consecutivet=int(ddt[2])

if args.scd not in ['misc','mv','none']:
    raise ValueError('scd must be misc, mv or none.')
thscd1,thscd2=args.thscd.split(',')

model_ver_nkv={2: 4,
               2.3: 5,
               2.4: 6,
               3.0: 7,
               3.1: 8,
               4: 9,
               4.1: 11,
               4.2: 13,
               4.3: 15,
               4.4: 17,
               4.5: 19,
               4.6: 21}
if args.model in model_ver_nkv:
    args.model = model_ver_nkv[args.model]
else:
    args.model=9
    
if args.model>=9:
    args.model+=args.slower_model

tmpV=os.path.abspath(tmpFolder+'_tmp.mkv') if args.start_time!=None or args.end_time!=None else inFile
tmpTSV2O=os.path.abspath(f'{tmpFolder}tsv2o.txt')
tmpTSV2N=os.path.abspath(f'{tmpFolder}tsv2nX8.txt')

ffpath=mmgpath=dllpath=toolsFolder
vspipepath=toolsFolder+'python-vapoursynth-plugins\\'

def processInfo():
    with open(tmpFolder+'infos.txt','r') as f:
        lines=[i.split('\t') for i in f][1:]
    for i in range(len(lines)):
        lines[i][0]=int(lines[i][0])
        lines[i][1]=int(float(lines[i][1])*1000)
        lines[i][2]=float(lines[i][2])
        lines[i][3]=int(lines[i][3])
    lines.sort()
    startpts=lines[0][1]
    dels=open(tmpFolder+'framestodelete.txt','w')
    tsv2o=open(tmpFolder+'tsv2o.txt','w')
    print('#timestamp format v2',file=tsv2o)
    consecutive=0
    for i in range(len(lines)):
        l=lines[i]
        if l[2]>=ssimt and l[3]<=pxdifft and consecutive<consecutivet:
            consecutive+=1
            print(l[0],file=dels)
        else:
            consecutive=0
            print(l[1]-startpts,file=tsv2o)
    dels.close()
    tsv2o.close()

def newTSgen():
    ts_new=list()
    outfile=open(tmpTSV2N,'w',encoding='utf-8')
    f=open(tmpTSV2O,'r',encoding='utf-8')
    ts_o=[i for i in f][1:]
    #print(ts_o)
    
    for x in range(len(ts_o)-1):
        ts_new.append(str(float(ts_o[x])))
        for i in range(1,8):
            ts_new.append( str(float(ts_o[x]) + (float(ts_o[x+1])-float(ts_o[x]))/8*i) )
    #print(ts_new)
    print('#timestamp format v2',file=outfile)
    for x in range(len(ts_new)):
        print(ts_new[x],file=outfile)
    print(ts_o[len(ts_o)-1],file=outfile)
    f.close()
    outfile.close()

def vpyGen():
    script_parse='''import vapoursynth as vs
core=vs.core
core.num_threads=%d
core.max_cache_size=%d
import xvs
clip = core.ffms2.Source(r"%s",cachefile="ffindex")
offs1 = core.std.BlankClip(clip,length=1)+clip[:-1]
offs1 = core.std.CopyFrameProps(offs1,clip)
offs1 = core.vmaf.Metric(clip,offs1,2)
offs1 = core.std.MakeDiff(offs1,clip)
offs1 = core.fmtc.bitdepth(offs1,bits=16)
offs1 = core.std.Expr(offs1,'x 32768 - abs')
offs1 = core.std.PlaneStats(offs1)
offs1 = xvs.props2csv(offs1,props=['_AbsoluteTime','float_ssim','PlaneStatsMax'],output='infos_running.txt',titles=[])
offs1.set_output()''' % (threads,args.maxmem,tmpV)

    scd=f'sup = core.mv.Super(clip,pel=1,levels=1)\nbw = core.mv.Analyse(sup,isb=True,levels=1,truemotion=False)\nclip = core.mv.SCDetection(clip,bw,thscd1={thscd1},thscd2={thscd2})' if args.scd=='mv' else 'clip = core.misc.SCDetect(clip)' if args.scd=='misc' else ''
    interp=\
    '''clip = core.rife.RIFE(clip,model={MVer},sc=True)
clip = core.rife.RIFE(clip,model={MVer},sc=True,uhd=True)
clip = core.rife.RIFE(clip,model={MVer},sc=True,uhd=True)'''.format(MVer=int(args.model)) \
    if args.model!=9 else \
    '''clip = core.rife.RIFE(clip,model=9,sc=True,factor_num=8,factor_den=1)'''

    script='''import vapoursynth as vs
core=vs.core
core.num_threads=%d
core.max_cache_size=%d
with open(r"framestodelete.txt","r") as f:
    dels=[int(i) for i in f]
clip = core.ffms2.Source(r"%s",cachefile="ffindex")
clip = core.std.DeleteFrames(clip,dels)
%s
clip = core.resize.Bicubic(clip,format=vs.RGBS,matrix_in=1)
%s
clip = core.resize.Bicubic(clip,format=vs.YUV420P10,matrix=1,dither_type='error_diffusion')
clip = core.vfrtocfr.VFRToCFR(clip,r"tsv2nX8.txt",%s,True)
sup = core.mv.Super(clip)
fw = core.mv.Analyse(sup)
bw = core.mv.Analyse(sup,isb=True)
clip = core.mv.FlowFPS(clip,sup,bw,fw,60,1)
clip.set_output()
''' % (threads,args.maxmem,tmpV,scd,interp,args.mf)

    with open(f'{tmpFolder}parse.vpy','w',encoding='utf-8') as vpy:
        print(script_parse,file=vpy)

    with open(f'{tmpFolder}interpX8.vpy','w',encoding='utf-8') as vpy:
        print(script,file=vpy)

if not os.path.exists(inFile):
    print('input file isn\'t exist')
    sys.exit()
if os.path.exists(outFile):
    if input('output file exists, continue? (y/n) ')=='y':
        pass
    else:
        sys.exit()
if os.path.exists(tmpFolder):
    if input('temp folder exists, continue? (y/n) ')=='y':
        pass
    else:
        sys.exit()
else:
    os.mkdir(tmpFolder)

if not os.path.exists(tmpV):
    ff_intermedia=f'\"{ffpath}ffmpeg.exe\" {ffss} {ffto} -i \"{inFile}\" -map 0:v:0 {ffau} {clo} -c:a flac -c:v copy -y \"{tmpFolder}cut.mkv\"'
    print(ff_intermedia)
    subprocess.run(ff_intermedia,shell=True)
    os.rename(f'{tmpFolder}cut.mkv',tmpV)

vpyGen()
if not os.path.exists(tmpFolder+'infos.txt'):
    print('calculating ssim.')
    parse=subprocess.run(f'\"{vspipepath}vspipe.exe\" \"{tmpFolder}parse.vpy\" -p .',shell=True)
    if parse.returncode==0:
        os.rename(tmpFolder+'infos_running.txt',tmpFolder+'infos.txt')
    else:
        raise RuntimeError('ssim parsing filed, please tyr again.')
processInfo()
newTSgen()
cmdinterp=f'\"{vspipepath}vspipe.exe\" -c y4m \"{tmpFolder}interpX8.vpy\" - | \"{ffpath}ffmpeg.exe\" -i - -i \"{tmpV}\" -map 0:v:0 {ffau2} -crf {crfo} {codecov} {codecoa} {abo} {ffparamo} \"{outFile}\" -y'
print(cmdinterp)
subprocess.run(cmdinterp,shell=True)
