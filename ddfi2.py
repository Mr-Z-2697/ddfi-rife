import os,sys
import argparse
import subprocess
import pathlib

toolsFolder=pathlib.Path(__file__).absolute().parent/'tools'
sys.path.append(str(toolsFolder))
class args:
    pass
parser = argparse.ArgumentParser(description='an video auto duplicated frame remove and frame interpolate script, uses ffmpeg and vapoursynth.',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i','--input',required=True,type=str,help='source file, any format ffmpeg can decode\n ')
parser.add_argument('-o','--output',required=False,type=str,help='output file, default \"input file\"_interp.mkv\n ')
parser.add_argument('-tf','--temp-folder',required=False,type=str,help='temp folder, default \"output file\"_tmp\\\n ')
parser.add_argument('-st','--start-time',required=False,type=str,help='cut input video start from this time, format h:mm:ss.nnn or seconds\n ')
parser.add_argument('-et','--end-time',required=False,type=str,help='cut input video end at this time, format h:mm:ss.nnn or seconds\n ')
parser.add_argument('-as','--audio-stream',required=False,type=str,help='set audio stream index, starts from 0, \n\"no\" means don\'t output audio, and is the default\n ',default='no')
parser.add_argument('-q','--output-crf',required=False,type=int,help='output video crf value, interger. if a codec don\'t has -crf option is used, \n--ffmpeg_params_output can be used as a workaround. default 18\n ')
parser.add_argument('-vc','--video-codec',required=False,type=str,help='output video codec, use the name in ffmpeg, \nwill be constrained by output format. default libx265\nthis arg can also set extra x265-params by use prefix "+"\n(e.g. +bframes=11:qpmax=45:qpmin=14)\n ')
parser.add_argument('-ac','--audio-codec',required=False,type=str,help='output audio codec, similar to -vc. default libopus\n ')
parser.add_argument('-al','--audio-channel-layout',required=False,type=str,help='output audio channel layout, \nuse the name in ffmpeg. default stereo\n ')
parser.add_argument('-ab','--audio-bitrate',required=False,type=str,help='output audio bitrate, use it like in ffmpeg. default 128k\n ')
parser.add_argument('-ddt','--dedup-thresholds',required=False,type=str,help='ssim, max pixel diff (16 bits scale)\nand max consecutive deletion, inclusive. \ndefault 0.999,10240,2\n ',default='0.999,10240,2')
parser.add_argument('--ffmpeg-params-output',required=False,type=str,help='other ffmpeg parameters used in final step, \nuse it carefully. default \"-map_metadata -1 -map_chapters -1\"\n ')
parser.add_argument('--scd',required=False,type=str,help='scene change detect method, \"misc\", \"mv\", \"sudo\" or \"none\", default mv\n ',default='mv')
parser.add_argument('--thscd',required=False,type=str,help='thscd1&2 of core.mv.SCDetection or thresh of sudo,\ndefault 200,85 if mv, 0.92 if sudo\n ',default=None)
parser.add_argument('--threads',required=False,type=int,help='how many threads to use in VS (core.num_threads),\ndefault auto detect (half of your total threads)\n ',default=None)
parser.add_argument('--maxmem',required=False,type=int,help='max memory to use for cache in VS (core.max_cache_size) in MB, default 4096\n ',default=4096)
parser.add_argument('-m','--model',required=False,type=str,help='model version, default 4.15\n ',default='4.15')
parser.add_argument('--slower-model',required=False,help='use ensemble model, some model won\'t work\ndefault false\n ',action=argparse.BooleanOptionalAction,default=False)
parser.add_argument('--vs-mlrt',required=False,help='use vs-mlrt, default false\n ',action=argparse.BooleanOptionalAction,default=False)
parser.add_argument('--mlrt-be',required=False,type=str,help='backend in vs-mlrt, default TRT\n ',default='TRT')
parser.add_argument('--mlrt-ns',required=False,type=int,help='num_streams in vs-mlrt, default 2\n ',default=2)
parser.add_argument('--mlrt-fp16',required=False,help='whether to use fp16 or not, default true\n ',action=argparse.BooleanOptionalAction,default=True)
parser.add_argument('--multi',required=False,type=int,help='multiple of interpolation, default 8\n ',default=8)
parser.add_argument('-mf','--medium-fps',required=False,type=str,help='medium fps, format is "fpsnum,fpsden", default 192000,1001\n ',default="192000,1001")
parser.add_argument('-of','--output-fps',required=False,type=str,help='output fps, format is "fpsnum,fpsden", default 60,1\n ',default="60,1")
parser.add_argument('--fast-fps-convert-down',required=False,help='use "fast mode" in the final fps convert down, default true\n ',action=argparse.BooleanOptionalAction,default=True)
parser.add_argument('--skip-encode',required=False,help='skip final output encoding, hence you can do it yourself or even play it directly\ndefault false\n ',action=argparse.BooleanOptionalAction,default=False)
parser.add_argument('--half-ssim',required=False,help='use 0.5x frame for ssim calculation, for speed, default true\n ',action=argparse.BooleanOptionalAction,default=True)
parser.add_argument('-impl','--mlrt-rife-impl',required=False,type=int,help='mlrt rife implementation, 1 or 2, default 1.',default=1)
parser.add_argument('-opt','--trt-optim-level',required=False,type=int,help='trt optimization level, 0-5, default 5.',default=5)
parser.add_argument('--adjacent',required=False,type=str,help=argparse.SUPPRESS,default='')#'delete adjacent frames of duplicated frames,\nthis can break consecutive deletion limit because of my garbage code,\nconsider this as for test purpose (string of relative frames like "+1,-1")\n '
parser.parse_args(sys.argv[1:],args)


inFile=pathlib.Path(args.input)
outFile=inFile.parent/(inFile.stem+'_interp.mkv') if args.output is None else pathlib.Path(args.output)
tmpFolder=outFile.parent/(outFile.stem+'_tmp') if args.temp_folder is None else pathlib.Path(args.temp_folder)

inFile,outFile,tmpFolder=map(pathlib.Path.absolute,(inFile,outFile,tmpFolder))
# tmpFolder+='\\' if tmpFolder[-1] != '\\' else ''

ffss='' if args.start_time is None else f'-ss {args.start_time}'
ffto='' if args.end_time is None else f'-to {args.end_time}'
ffau='-an' if args.audio_stream == 'no' else f'-map a:{args.audio_stream}'
ffau2='-an' if args.audio_stream == 'no' else f'-map 1:a:0'
crfo=18 if args.output_crf is None else args.output_crf
x265_default='-c:v libx265 -preset 6 -x265-params sao=0:rect=0:strong-intra-smoothing=0:open-gop=0:aq-mode=1:aq-strength=0.8:ctu=32:rc-lookahead=60:me=hex:subme=2'
codecov=x265_default if args.video_codec is None else x265_default+':'+args.video_codec[1:] if args.video_codec[0]=='+' else f'-c:v {args.video_codec}'
codecoa='-c:a libopus' if args.audio_codec is None else f'-c:a {args.audio_codec}'
clo='-channel_layout stereo' if args.audio_channel_layout is None else f'-channel_layout {args.audio_channel_layout}'
abo='-b:a 128k' if args.audio_bitrate is None else f'-b:a {args.audio_bitrate}'
ffparamo='-map_metadata -1 -map_chapters -1' if args.ffmpeg_params_output is None else args.ffmpeg_params_output
threads='core.num_threads//2' if args.threads is None else args.threads
ddt=args.dedup_thresholds.split(',')
ssimt=float(ddt[0])
pxdifft=int(ddt[1])
consecutivet=int(ddt[2])

if args.scd not in ['misc','mv','sudo','none']:
    raise ValueError('scd must be misc, mv, sudo or none.')
if args.thscd==None:
    if args.scd=='mv':
        thscd1,thscd2=200,85
    if args.scd=='sudo':
        thscd=0.92
else:
    if args.scd=='mv':
        thscd1,thscd2=args.thscd.split(',')
    if args.scd=='sudo':
        thscd=args.thscd

if args.scd=='sudo':
    sudo_onnx=(toolsFolder/'scd-model').glob('*.onnx')
    sudo_onnx=list(sudo_onnx)
    if len(sudo_onnx)!=1:
        raise RuntimeError('exactly one sudo onnx model required.')
    sudo_onnx=sudo_onnx[0]

model_ver_nvk={'2': 4,
               '2.3': 5,
               '2.4': 6,
               '3.0': 7,
               '3.1': 8,
               '3.9': 9,
               '4.0': 11,
               '4.1': 13,
               '4.2': 15,
               '4.3': 17,
               '4.4': 19,
               '4.5': 21,
               '4.6': 23,
               '4.7': 25,
               '4.8': 27,
               '4.9': 29,
               '4.10':31,
               '4.11':33,
               '4.12':35,
               '4.12-lite':37,
               '4.13':39,
               '4.13-lite':41,
               '4.14':43,
               '4.14-lite':45,
               '4.15':47,
               '4.15-lite':49,
               '4.16-lite':51, # deprecated?
               '4.17':53,
               }
model_ver_mlrt={'4':40,
                '4.0':40,
                '4.2':42,
                '4.3':43,
                '4.4':44,
                '4.5':45,
                '4.6':46,
                '4.7':47,
                '4.8':48,
                '4.9':49,
                '4.10':410,
                '4.11':411,
                '4.12':412,
                '4.12-lite':4121,
                '4.13':413,
                '4.13-lite':4131,
                '4.14':414,
                '4.14-lite':4141,
                '4.15':415,
                '4.15-lite':4151,
                '4.16-lite':4161, # deprecated?
                '4.17':417,
                }
if not args.vs_mlrt:
    if args.model in model_ver_nvk:
        args.model = model_ver_nvk[args.model]
    else:
        args.model=24

    if args.model>=9 and args.model<54:
        args.model+=args.slower_model
else:
    if args.model in model_ver_mlrt:
        args.model = model_ver_mlrt[args.model]
    elif args.model in model_ver_mlrt.values():
        pass
    else:
        args.model = 48

tmpV=tmpFolder/'_tmp.mkv' if args.start_time!=None or args.end_time!=None else inFile
tmpParseVpy=tmpFolder/'parse.vpy'
tmpInterpVpy=tmpFolder/f'interpX{args.multi}.vpy'
tmpInfos=tmpFolder/'infos.txt'
tmpDelList=tmpFolder/'framestodelete.txt'
tmpTSV2O=tmpFolder/'tsv2o.txt'
tmpTSV2N=tmpFolder/f'tsv2nX{args.multi}.txt'

ffpath=toolsFolder/'ffmpeg'
vspipepath=toolsFolder/'python-vapoursynth-plugins'/'vspipe'

def processInfo():
    with open(tmpInfos,'r') as f:
        lines=[i.split('\t') for i in f][1:]
    for i in range(len(lines)):
        lines[i][0]=int(lines[i][0])
        lines[i][1]=int(float(lines[i][1])*1000)
        lines[i][2]=float(lines[i][2])
        lines[i][3]=int(lines[i][3])
    lines.sort()
    startpts=lines[0][1]
    lastframe=lines[-1][0]
    consecutive=0
    adjacents=[int(i) for i in args.adjacent.split(',')] if args.adjacent else []
    del_list=[]
    tsv2_list=[]
    for i in range(1,len(lines)-1):
        l=lines[i]
        if l[2]>=ssimt and l[3]<=pxdifft and consecutive<consecutivet and not l[0] in del_list:
            consecutive+=1
            del_list.append(l[0])
            for i in adjacents:
                fadj=l[0]+i
                if 0<fadj<lastframe and not fadj in del_list:
                    consecutive+=1
                    del_list.append(fadj)
        elif not l[0] in del_list:
            consecutive=0
    for i in range(len(lines)):
            l=lines[i]
            if not l[0] in del_list:
                tsv2_list.append(l[1]-startpts)
    with open(tmpDelList,'w') as dels:
        dels.write('\n'.join(map(str,del_list)))
    with open(tmpTSV2O,'w') as tsv2o:
        tsv2o.write('#timestamp format v2\n')
        tsv2o.write('\n'.join(map(str,tsv2_list)))

def newTSgen():
    ts_new=list()
    outfile=open(tmpTSV2N,'w',encoding='utf-8')
    f=open(tmpTSV2O,'r',encoding='utf-8')
    ts_o=[i for i in f][1:]
    #print(ts_o)
    
    for x in range(len(ts_o)-1):
        ts_new.append(str(float(ts_o[x])))
        for i in range(1,args.multi):
            ts_new.append( str(float(ts_o[x]) + (float(ts_o[x+1])-float(ts_o[x]))/args.multi*i) )
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
core.num_threads={NT}
core.max_cache_size={MCS}
import xvs
from math import floor
clip = core.lsmas.LWLibavSource(r"{SRC}",cachefile="lwindex")
halfw,halfh = floor(clip.width/4)*2,floor(clip.height/4)*2
clip1 = core.resize.Bilinear(clip,halfw,halfh)
offs1 = core.std.BlankClip(clip1,length=1)+clip1[:-1]
offs1 = core.std.CopyFrameProps(offs1,clip1)
ssim = core.vmaf.Metric(clip1,offs1,2)
offs1 = core.std.BlankClip(clip,length=1)+clip[:-1]
offs1 = core.std.CopyFrameProps(offs1,ssim)
offs1 = core.std.MakeDiff(offs1,clip)
offs1 = core.fmtc.bitdepth(offs1,bits=16)
offs1 = core.std.Expr(offs1,'x 32768 - abs')
offs1 = core.std.PlaneStats(offs1)
offs1 = xvs.props2csv(offs1,props=['_AbsoluteTime','float_ssim','PlaneStatsMax'],output='infos_running.txt',titles=[])
offs1.set_output()''' \
    if args.half_ssim else \
        '''import vapoursynth as vs
core=vs.core
core.num_threads={NT}
core.max_cache_size={MCS}
import xvs
clip = core.lsmas.LWLibavSource(r"{SRC}",cachefile="lwindex")
offs1 = core.std.BlankClip(clip,length=1)+clip[:-1]
offs1 = core.std.CopyFrameProps(offs1,clip)
offs1 = core.vmaf.Metric(clip,offs1,2)
offs1 = core.std.Expr([offs1,clip],'x y - abs').fmtc.bitdepth(bits=16,dmode=1)
offs1 = core.std.PlaneStats(offs1)
offs1 = xvs.props2csv(offs1,props=['_AbsoluteTime','float_ssim','PlaneStatsMax'],output='infos_running.txt',titles=[])
offs1.set_output()'''
    script_parse=script_parse.format(NT=threads,MCS=args.maxmem,SRC=tmpV)

    if args.scd=='mv':
        scd=f'sup = core.mv.Super(clip,pel=1,levels=1)\nbw = core.mv.Analyse(sup,isb=True,levels=1,truemotion=False)\nclip = core.mv.SCDetection(clip,bw,thscd1={thscd1},thscd2={thscd2})'
    elif args.scd=='sudo':
        scd=f'import scene_detect as scd\nclip = scd.scene_detect(clip,onnx_path=r"{sudo_onnx}",thresh={thscd})'
    elif args.scd=='misc':
        scd='clip = core.misc.SCDetect(clip)'
    else:
        scd=''

    if not args.vs_mlrt:
        interp=\
            '''clip = core.rife.RIFE(clip,model={MVer},sc=True)
clip = core.rife.RIFE(clip,model={MVer},sc=True,uhd=True)
clip = core.rife.RIFE(clip,model={MVer},sc=True,uhd=True)'''.format(MVer=int(args.model)) \
        if args.model<9 else \
            '''clip = core.rife.RIFE(clip,model={MVer},sc=True,factor_num={MUL},factor_den=1)'''.format(MVer=int(args.model),MUL=args.multi)
    elif args.mlrt_rife_impl==2:
        interp=\
            '''from vsmlrt import RIFE,Backend
clip = core.resize.Bicubic(clip,matrix_in=matrix,format=vs.RGB{HS})
clip = RIFE(clip,model={MVer},ensemble={ENSE},multi={MUL},backend=Backend.{BE}(num_streams={NS},fp16={FP16}{OFMT}{OPTIM}),_implementation=2)
clip = core.resize.Bicubic(clip,matrix=matrix,format=src_fmt.replace(bits_per_sample=10),dither_type='ordered')'''.format(MVer=int(args.model),MUL=args.multi,BE=args.mlrt_be,NS=args.mlrt_ns,FP16=args.mlrt_fp16,HS='SH'[args.mlrt_fp16],OFMT=',output_format='+str(int(args.mlrt_fp16)) if args.mlrt_be=='TRT' else '',ENSE=args.slower_model,OPTIM=f',builder_optimization_level={args.trt_optim_level}' if args.mlrt_be=='TRT' else '')
    else:
        interp=\
            '''from vsmlrt import RIFE,Backend
from math import ceil
src_w = clip.width
src_h = clip.height
pad_w = ceil(src_w/32)*32
pad_h = ceil(src_h/32)*32
clip = core.resize.Bicubic(clip,pad_w,pad_h,src_width=pad_w,src_height=pad_h,matrix_in=matrix,format=vs.RGB{HS})
clip = RIFE(clip,model={MVer},ensemble={ENSE},multi={MUL},backend=Backend.{BE}(num_streams={NS},fp16={FP16}{OFMT}{OPTIM}),_implementation=1)
clip = core.resize.Bicubic(clip,src_w,src_h,src_width=src_w,src_height=src_h,matrix=matrix,format=src_fmt.replace(bits_per_sample=10),dither_type='ordered')'''.format(MVer=int(args.model),MUL=args.multi,BE=args.mlrt_be,NS=args.mlrt_ns,FP16=args.mlrt_fp16,HS='SH'[args.mlrt_fp16],OFMT=',output_format='+str(int(args.mlrt_fp16)) if args.mlrt_be=='TRT' else '',ENSE=args.slower_model,OPTIM=f',builder_optimization_level={args.trt_optim_level}' if args.mlrt_be=='TRT' else '')

    script='''import vapoursynth as vs
core=vs.core
core.num_threads={NT}
core.max_cache_size={MCS}
with open(r"framestodelete.txt","r") as f:
    dels=[int(i) for i in f]
clip = core.lsmas.LWLibavSource(r"{SRC}",cachefile="lwindex")
src_fmt=clip.format
try:
    matrix=clip.get_frame(0).props._Matrix
except:
    matrix=1
clip = core.std.DeleteFrames(clip,dels)
{SCD}
{TORGB}
{INT}
{TOYUV}
clip = core.vfrtocfr.VFRToCFR(clip,r"tsv2nX{MUL}.txt",{MF},True)
sup = core.mv.Super(clip){FAST}
fw = core.mv.Analyse(sup){FAST}
bw = core.mv.Analyse(sup,isb=True){FAST}
clip = core.mv.{XFPS}(clip,sup,bw,fw,{OF})
clip.set_output()
'''.format(NT=threads,MCS=args.maxmem,SRC=tmpV,SCD=scd,INT=interp,MF=args.medium_fps,OF=args.output_fps,MUL=args.multi,
TORGB='clip = core.resize.Bicubic(clip,format=vs.RGBS,matrix_in=matrix)' if not args.vs_mlrt else '',
TOYUV='clip = core.resize.Bicubic(clip,format=src_fmt.replace(bits_per_sample=10),matrix=matrix,dither_type="ordered")' if not args.vs_mlrt else '',
FAST='[0]*clip.num_frames' if args.fast_fps_convert_down else '',XFPS='BlockFPS' if args.fast_fps_convert_down else 'FlowFPS')

    with open(tmpParseVpy,'w',encoding='utf-8') as vpy:
        print(script_parse,file=vpy)

    with open(tmpInterpVpy,'w',encoding='utf-8') as vpy:
        print(script,file=vpy)

if not inFile.exists():
    print('input file isn\'t exist')
    sys.exit()
if outFile.exists():
    if input('output file exists, continue? (y/n) ')=='y':
        pass
    else:
        sys.exit()
if tmpFolder.exists():
    if input('temp folder exists, continue? (y/n) ')=='y':
        pass
    else:
        sys.exit()
else:
    os.mkdir(tmpFolder)

if not tmpV.exists():
    cutpath=tmpFolder/'cut.mkv'
    ff_intermedia=f'\"{ffpath}\" {ffss} {ffto} -i \"{inFile}\" -map 0:v:0 {ffau} {clo} -c:a flac -c:v copy -y \"{cutpath}\"'
    print(ff_intermedia)
    subprocess.run(ff_intermedia,shell=True)
    os.rename(cutpath,tmpV)

vpyGen()
if not tmpInfos.exists():
    print('calculating ssim.')
    parse=subprocess.run(f'\"{vspipepath}\" \"{tmpParseVpy}\" -p .',shell=True)
    if parse.returncode==0:
        os.rename(tmpFolder/'infos_running.txt',tmpInfos)
    else:
        raise RuntimeError('ssim parsing failed, please check your settings then try again.')
processInfo()
newTSgen()
cmdinterp=f'\"{vspipepath}\" -c y4m \"{tmpInterpVpy}\" - | \"{ffpath}\" -i - -i \"{tmpV}\" -map 0:v:0 {ffau2} -crf {crfo} {codecov} {codecoa} {abo} {ffparamo} \"{outFile}\" -y'
print(cmdinterp)
if not args.skip_encode:
    subprocess.run(cmdinterp,shell=True)
