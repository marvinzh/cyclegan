#!/usr/bin/perl
#
# Copyright   2013   Daniel Povey
# Apache 2.0

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC99S79> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora5/LDC/LDC99S79 data/swbd2_phase2_train\n";
  exit(1);
}
($db_base, $out_dir) = @ARGV;

if (system("mkdir -p $out_dir")) {
  die "Error making directory $out_dir";
}

open(CS, "<$db_base/doc/SWB_callstat.tbl") || die  "Could not open $db_base/DISC1/doc/callstat.tbl";
#open(CI, "<$db_base/DISC1/doc/callinfo.tbl") || die  "Could not open $db_base/DISC1/doc/callinfo.tbl";
open(GNDR, ">$out_dir/spk2gender") || die "Could not open the output file $out_dir/spk2gender";
open(SPKR, ">$out_dir/utt2spk") || die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">$out_dir/wav.scp") || die "Could not open the output file $out_dir/wav.scp";

@badAudio = ("3", "4");

$tmp_dir = "$out_base/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

if (system("find $db_base -name '*.sph' > $tmp_dir/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(WAVLIST, "<", "$tmp_dir/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $uttId=$t1[0];
  $wav{$uttId} = $sph;
}


while (<CS>) {
  $line = $_ ;
  @A = split(",", $line);
  $wav = $A[1];
  if (/$wav/i ~~ @badAudio) {
    # do nothing 
  } else {
    $spkr1 = "sp2_" . $A[0];
    $spkch1 = $A[3];
    if ($spkch1 eq "1\n") {
      $spkch1 = "1";
    } elsif ($spkch1 eq "2\n") {
      $spkch1 = "2";
    } else {
      die "Unknown Channel in $line";
    }
    $gender1 = $A[2];
    if ($gender1 eq "m") {
      $gender1 = "m";
    } elsif ($gender1 eq "f") {
      $gender1 = "f";
    } else {
      die "Unknown Gender in $line";
    }

    if (-e "$wav{$wav}") {
      $uttId = $spkr1 ."_" . $wav ."_$spkch1";
      if (!$spk2gender{$spkr1}) {
        $spk2gender{$spkr1} = $gender1;
        print GNDR "$spkr1"," $gender1\n";
      }
      print WAV "$uttId"," sph2pipe -f wav -p -c $spkch1 $wav{$wav} |\n";
      print SPKR "$uttId"," $spkr1","\n";
    } else {
      print STDERR "Missing $wav{$wav} for $wav\n";
    }
  }
}


close(WAV) || die;
close(SPKR) || die;
close(GNDR) || die;
if (system("utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
if (system("utils/fix_data_dir.sh $out_dir") != 0) {
  die "Error fixing data dir $out_dir";
}
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
