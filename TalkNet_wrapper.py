import os, cv2, numpy, subprocess, tqdm, glob, sys, math
import pickle as pkl 
from scipy import signal
from scipy.io import wavfile
from TalkNet.talkNet import talkNet
from src.local_utils import writeToPickleFile
import python_speech_features
import torch
import numpy as np

class TalkNetWrapper():
	def __init__(self, videoPath, cacheDir, framesObj=None):
		self.videoPath = videoPath
		self.cacheDir = cacheDir
		self.cropScale = 0.25
		self.audioFilePath = os.path.join(self.cacheDir, 'audio.wav')
		self.nDataLoaderThread = 10
		self.pretrainModel = '../TalkNet/pretrain_TalkSet.model'
		if not framesObj:
			frameFilePath = os.path.join(self.cacheDir, 'frames.pkl')
			self.framesObj = pkl.load(open(frameFilePath, 'rb'))
		else:
			self.framesObj = framesObj
		if os.path.isfile(self.pretrainModel) == False: # Download the pretrained model
			Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
			cmd = "gdown --id %s -O %s"%(Link, self.pretrainModel)
			subprocess.call(cmd, shell=True, stdout=None)

	def readFaceTracks(self, faceTracks=None):
		if not faceTracks:
			faceTracksFile = os.path.join(self.cacheDir, 'face_retinaFace.pkl')
			faceTracks = pkl.load(open(faceTracksFile, 'rb'))
		allTracks = []
		for faceTrackId, faceTrack in faceTracks.items():
			frameNums = [int(round(face[0]*self.framesObj['fps'])) for face in faceTrack]
			boxes = []
			for box in faceTrack:
				x1 = int(round(box[1]*self.framesObj['width']))
				y1 = int(round(box[2]*self.framesObj['height']))
				x2 = int(round(box[3]*self.framesObj['width']))
				y2 = int(round(box[4]*self.framesObj['height']))
				boxes.append([x1, y1, x2, y2])
			allTracks.append({'frame':frameNums, 'bbox':boxes, 'trackId': faceTrackId})
		return allTracks

	def crop_video(self, track, cropFile):
		# CPU: crop the face clips
		allFrames = self.framesObj['frames']
		dets = {'x':[], 'y':[], 's':[]}
		for det in track['bbox']: # Read the tracks
			dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
			dets['y'].append((det[1]+det[3])/2) # crop center x 
			dets['x'].append((det[0]+det[2])/2) # crop center y
		dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
		dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
		dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
		if os.path.isfile(f'{cropFile}.avi'):
			return {'trackId': cropFile.split('/')[-1], 'track':track, 'proc_track':dets}
		vOut = cv2.VideoWriter(
			f'{cropFile}t.avi', cv2.VideoWriter_fourcc(*'XVID'), self.framesObj['fps'], (224, 224)
		)
		for fidx, frame in enumerate(track['frame']):
			cs  = self.cropScale
			bs  = dets['s'][fidx]   # Detection box size
			bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
			image = allFrames[frame]
			frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
			my  = dets['y'][fidx] + bsi  # BBox center Y
			mx  = dets['x'][fidx] + bsi  # BBox center X
			face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
			if (int(my+bs*(1+2*cs)) - int(my-bs) > 0) and (int(mx+bs*(1+cs)) - int(mx-bs*(1+cs)) > 0):
				out_face = face
			else:
				continue
			vOut.write(cv2.resize(out_face, (224, 224)))
		
		assert self.framesObj['fps'] == 25, f'fps is not 25 but {self.framesObj["fps"]}'
		audioTmp = f'{cropFile}.wav'
		audioStart  = (track['frame'][0]) / self.framesObj['fps']
		audioEnd    = (track['frame'][-1]+1) / self.framesObj['fps']
		vOut.release()
		#extract the audio file
		command = ("ffmpeg -y -nostdin -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
						(self.audioFilePath, self.nDataLoaderThread, audioStart, audioEnd, audioTmp))
		output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
		_, audio = wavfile.read(audioTmp)
		command = ("ffmpeg -y -nostdin -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
						(cropFile, audioTmp, self.nDataLoaderThread, cropFile)) # Combine audio and video file
		output = subprocess.call(command, shell=True, stdout=None)
		os.remove(f'{cropFile}t.avi')
		# convert to 25fps
		# command = f'ffmpeg -y -nostdin -loglevel panic -i {cropFile}tt.avi -filter:v fps=25 {cropFile}.avi'
		# output = subprocess.call(command, shell=True, stdout=None)
		# os.remove(f'{cropFile}tt.avi')
		# #extract the audio file from 25fps
		# command = f'ffmpeg -y -nostdin -loglevel error -i {cropFile}.avi \
		#         -ar 16k -ac 1 {cropFile}.wav'
		# output = subprocess.call(command, shell=True, stdout=None)
		return {'trackId': cropFile.split('/')[-1], 'track':track, 'proc_track':dets}

	def evaluate_network(self, files):
		# GPU: active speaker detection by pretrained TalkNet
		s = talkNet()
		s.loadParameters(self.pretrainModel)
		sys.stderr.write("Model %s loaded from previous state! \r\n"%self.pretrainModel)
		s.eval()
		allScores = []
		# durationSet = {1,2,4,6} # To make the result more reliable
		durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
		for file in tqdm.tqdm(files, total = len(files)):
			audioPath = file + '.wav' # Load audio and video
			_, audio = wavfile.read(audioPath)
			audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
			video = cv2.VideoCapture(file + '.avi')
			# check for the length of the video
			if video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS) < 0.4:
				allScores.append('nan')
				continue
			videoFeature = []
			while video.isOpened():
				ret, frames = video.read()
				if ret == True:
					face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
					face = cv2.resize(face, (224,224))
					face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
					videoFeature.append(face)
				else:
					break
			video.release()
			videoFeature = numpy.array(videoFeature)
			length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100.0, videoFeature.shape[0]/ 25.0)
			audioFeature = audioFeature[:int(round(length * 100)),:]
			videoFeature = videoFeature[:int(round(length * 25)),:,:]
			allScore = [] # Evaluation use TalkNet
			for duration in durationSet:
				batchSize = int(math.ceil(length / duration))
				scores = []
				with torch.no_grad():
					for i in range(batchSize):
						inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
						inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
						embedA = s.model.forward_audio_frontend(inputA)
						embedV = s.model.forward_visual_frontend(inputV)
						embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
						out = s.model.forward_audio_visual_backend(embedA, embedV)
						score = s.lossAV.forward(out, labels = None)
						scores.extend(score)
				allScore.append(scores)
			allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
			allScores.append(allScore)	
		return allScores

	def visualization(self, tracks, scores):
		# CPU: visulize the result for video format
		flist = self.framesObj['frames']
		faces = [[] for i in range(len(flist))]
		for tidx, track in enumerate(tracks):
			score = scores[tidx]
			for fidx, frame in enumerate(track['track']['frame']):
				s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
				s = numpy.mean(s)
				faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
		# firstImage = cv2.imread(flist[0])
		fw = self.framesObj['width']
		fh = self.framesObj['height']
		vOut = cv2.VideoWriter(os.path.join(self.cacheDir, 'talknet_video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
		colorDict = {0: 0, 1: 255}
		for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
			image = fname
			for face in faces[fidx]:
				box = [int(face['x']-face['s'])/fw, int(face['y']-face['s'])/fh,\
						int(face['x']+face['s'])/fw, int(face['y']+face['s'])/fh]
				clr = colorDict[int((face['score'] >= 0))]
				txt = round(face['score'], 1)
				cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
				cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
			vOut.write(image)
		vOut.release()
		command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
			(os.path.join(self.cacheDir, 'talknet_video_only.avi'), os.path.join(self.cacheDir, 'audio.wav'), \
			self.nDataLoaderThread, os.path.join(self.cacheDir,'talknet_video_out.avi'))) 
		output = subprocess.call(command, shell=True, stdout=None)

	def faceWiseScores(self, tracks, scores):
		faceScores = {}
		for tidx, track in enumerate(tracks):
			score = scores[tidx]
			if score == 'nan':
				track_scores = ['nan']*len(track)
				faceScores[track['trackId']] = track_scores
				continue
			track_scores = []
			for fidx, frame in enumerate(track['track']['frame']):
				s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
				s = numpy.mean(s)
				track_scores.append(s)
			faceScores[track['trackId']] = track_scores
		return faceScores

	def run_visualization(self, scores):
		frames = self.framesObj['frames']
		faceTracksFile = os.path.join(self.cacheDir, 'face_retinaFace.pkl')
		faceTracks = pkl.load(open(faceTracksFile, 'rb'))
		colorDict = {0: 0, 1: 255}
		for faceTrackId, faceTrack in faceTracks.items():
			for box, score in zip(faceTrack, scores[faceTrackId]):
				frameNo = int(round(box[0]*self.framesObj['fps']))
				if frameNo < len(frames):
					x1 = int(round(box[1]*self.framesObj['width']))
					y1 = int(round(box[2]*self.framesObj['height']))
					x2 = int(round(box[3]*self.framesObj['width']))
					y2 = int(round(box[4]*self.framesObj['height']))
					clr = colorDict[int((score >= 0))] if not np.isnan(float(score)) else 0
					cv2.rectangle(frames[frameNo], (x1, y1), (x2, y2), (0,clr,255-clr), 2)
					# printing the name of the face track
					cv2.putText(frames[frameNo], f'{score:.2f}', (x1, y1 - 10), \
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		videoSavePath = os.path.join(self.cacheDir, 'talknet_video_out.mp4')
		video_writer = cv2.VideoWriter(videoSavePath, cv2.VideoWriter_fourcc(*'mp4v'), \
								   self.framesObj['fps'], (int(self.framesObj['width']), int(self.framesObj['height'])))
		for frame in frames:
			video_writer.write(frame)
		video_writer.release()
		videoSavePathTmp = os.path.join(self.cacheDir, f'talknet_video_out_tmp.mp4')
		wavPath = os.path.join(self.cacheDir, 'audio.wav')
		audio_video_merge_cmd  = f'ffmpeg -loglevel error -i {videoSavePath} -i {wavPath} -c:v copy -c:a aac {videoSavePathTmp}'
		subprocess.call(audio_video_merge_cmd, shell=True, stdout=False)
		os.rename(f'{videoSavePathTmp}', videoSavePath)
		print(f'talknet asd video saved at {videoSavePath}')

	def run(self, faceTracks=None, visualization=False):
		talknetScoresFile = os.path.join(self.cacheDir, 'talknet_scores.pkl')
		if os.path.isfile(talknetScoresFile):
			print('reading talknet scores from cache')
			scores = pkl.load(open(talknetScoresFile, 'rb'))
			if visualization:
				self.run_visualization(scores)
			return scores

		print('computing talknet scores')
		# frameFilePath = os.path.join(self.cacheDir, 'frames.pkl')
		# self.framesObj = pkl.load(open(frameFilePath, 'rb'))
		faceCropDir = os.path.join(self.cacheDir, 'face_crop_videos')
		os.makedirs(faceCropDir, exist_ok=True)
		allTracks = self.readFaceTracks(faceTracks)
		vidTracks = [
			self.crop_video(
				track,
				os.path.join(faceCropDir, track['trackId']),
			)
			for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks), desc='video crops for TalkNet')
		]
		files = [os.path.join(faceCropDir, track['trackId']) for track in vidTracks]
		scores = self.evaluate_network(files)
		scores = self.faceWiseScores(vidTracks, scores)
		writeToPickleFile(scores, talknetScoresFile)
		if visualization:
			self.run_visualization(scores)
		return scores	
		

