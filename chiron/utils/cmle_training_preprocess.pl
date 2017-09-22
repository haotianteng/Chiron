#!/usr/bin/perl
use strict;
my $base = shift @ARGV;
my $out  = shift @ARGV;
if ( ! $base || ! $out ) {
  print STDERR "Usage: $0 <input path prefix> <output path>\nwhere appending prefix with (.signal,.label) are readable paths\n";
  exit(1);
}
open( SIGNAL, "$base.signal" ) or die "Can't open .signal file: $!";
open( LABEL,  "$base.label"  ) or die "Can't open .label  file: $!";
open( OUT,    ">$out"        ) or die "Can't write to '$out':   $!";
my @signal = split /\s+/, <SIGNAL>;
while ( my $line = <LABEL> ) {
  chomp $line;
  my ( $start, $end, $label ) = split /\s+/, $line;
  my @slice = @signal[ $start .. ($end-1) ];
  print OUT $label, "\t", $start, "\t", $end, "\t", scalar(@signal), "\t", join( ",", @slice ), "\t", $base, "\n";
}
