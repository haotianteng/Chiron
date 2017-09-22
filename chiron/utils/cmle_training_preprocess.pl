#!/usr/bin/perl
use strict;
my $base = shift @ARGV;

if ( ! $base ) {
  print STDERR "Usage: $0 [input prefix]\nwhere appending prefix with (.signal,.label) are readable paths\n";
  exit(1);
}

open( SIGNAL, "$base.signal" ) or die "Can't open .signal file: $!";
open( LABEL,  "$base.label" )  or die "Can't open .label  file: $!";
my @signal = split /\s+/, <SIGNAL>;

while ( my $line = <LABEL> ) {
  chomp $line;
  my ( $start, $end, $label ) = split /\s+/, $line;
  my @slice = @signal[ $start .. ($end-1) ];
  print $label, "\t", join( ",", @slice ), "\n";
}
