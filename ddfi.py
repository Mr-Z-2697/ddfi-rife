import os,sys
import argparse
import subprocess

class args:
    pass
parser = argparse.ArgumentParser(description='animation auto duplicated frame remove and frame interpolate tool',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i','--input',required=True,type=str,help='source file, any format ffmpeg can decode')
parser.add_argument('-o','--output',required=False,type=str,help='output file, default \"input file\"_interp.mkv')
parser.add_argument('-tf','--temp_folder',required=False,type=str,help='temp folder, default \"output file\"_tmp\\')
parser.add_argument('-st','--start_time',required=False,type=str,help='cut input video from this time, format h:mm:ss.nnn')
parser.add_argument('-et','--end_time',required=False,type=str,help='cut input video end to this time, format h:mm:ss.nnn')
parser.add_argument('-as','--audio_stream',required=True,type=str,help='set audio stream index, starts from 0, \"no\" means don\'t output audio')
parser.add_argument('-q','--output_crf',required=False,type=int,help='output video crf value, default 18')
parser.add_argument('-qi','--intermedia_crf',required=False,type=int,help='intermedia video crf value, default 12')
parser.add_argument('-if','--interpolation_factor',required=False,type=int,help='interpolation factor, default 8, not recommend to decrease it')
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

tmpDedup=os.path.abspath(tmpFolder+'dedup_done.mkv')
tmpInterpXn=f'{tmpFolder}interpX{xinterp}_done.264'
tmpTSV2O=f'{tmpFolder}tsv2o.txt'
tmpTSV2N=f'{tmpFolder}tsv2nX{xinterp}.txt'

ffpath=''
mmgpath='D:\\Softwares\\Mkvtoolnix\\'
dllpath=os.path.dirname(os.path.realpath(__file__))

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
    ff_mpdecimate=f'\"{ffpath}ffmpeg.exe\" {ffss} {ffto} -i \"{inFile}\" -vf mpdecimate=max=2 -crf {crfi} -preset 1 -pix_fmt yuv420p -sample_fmt s16 -c:v libx264 -c:a flac -map v:0 {ffau} \"{tmpFolder}dedup.mkv\" -y'
    print(ff_mpdecimate)
    subprocess.run(ff_mpdecimate,shell=True)
    os.rename(f'{tmpFolder}dedup.mkv',tmpDedup)

if not os.path.exists(tmpInterpXn):
    script=f'''LoadPlugin(\"{dllpath}\\LSMASHSource.dll\")
LoadPlugin(\"{dllpath}\\svpflow1.dll\")
LoadPlugin(\"{dllpath}\\svpflow2.dll\")
LWLibavVideoSource(\"{tmpDedup}\")
AssumeFPS(10)
'''
    script+='''
Threads=4
super_params="{pel:2,gpu:1}"
analyse_params="""{block:{w:16,h:16}, main:{search:{coarse:{distance:-10}}}, refine:[{thsad:200}]}""" 
smoothfps_params="{rate:{num:%d,den:2},algo:23,cubic:1}"
super = SVSuper(super_params)
vectors = SVAnalyse(super, analyse_params)
SVSmoothFps(super, vectors, smoothfps_params, mt=threads)
Prefetch(threads)
''' % (2*xinterp)
    avs=open(f'{tmpFolder}interpX{xinterp}.avs','w')
    print(script,file=avs)
    avs.close()
    ff_interp=f'\"{ffpath}ffmpeg\" -i \"{tmpFolder}interpX{xinterp}.avs\" -crf {crfi} -preset 1 -pix_fmt yuv420p -c:v libx264 -map v:0 \"{tmpFolder}interp.264\" -y'
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

ff_down60=f'\"{ffpath}ffmpeg.exe\" -i \"{tmpInterpXn}.mkv\" -i {tmpDedup} -map 0:v:0 {ffau2} -vf framerate=60000/1001 -crf {crfo} -pix_fmt yuv420p -c:v libx264 -c:a aac -b:a 256k \"{outFile}\" -y'
print(ff_down60)
subprocess.run(ff_down60,shell=True)
